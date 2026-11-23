"""
Acrobatic Multi-Loop Flight Simulation & Trajectory Renderer.

Executes the Drosophila connectome flight controller through the user's exact
42-waypoint cubic B-spline trajectory.
Features:
- Exact 42 control points matching user's reference image and script.
- Scipy cubic B-spline interpolation (splprep / splev, k=3).
- Continuous 360° tangent body heading (no backwards flight).
- Dynamic coordinated banking and pitch tilt ('bending over into loops').
- Anti-vibration haltere gyroscopic Coriolis filter + neuromuscular muscle smoothing.
- 3D cyan trajectory ribbon rendered directly in the camera scene with MuJoCo capsules.
- True 1.0x Real-Time 60 FPS HD MP4 video recording.
"""
import os
import sys
import time
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
from scipy.interpolate import splprep, splev

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

os.environ['MUJOCO_GL'] = 'egl'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import mujoco
import tensorflow as tf
from flybody.tasks.trajectory_loaders import InferenceFlightTrajectoryLoader
from orthocop.flybody.closed_loop import ClosedLoopFlyBodySim


def mult_quat(q1, q2):
    res = np.zeros(4)
    mujoco.mju_mulQuat(res, q1, q2)
    return res


# ==========================================================
# 1. DESIGN THE 3D TRAJECTORY (USER'S EXACT 42 CONTROL POINTS)
# ==========================================================
def get_user_control_points():
    """Exact 42 3D control points provided by user."""
    return np.array([
        [-9, -8,  3],
        [-7, -3,  9],
        [-3,  1, 15],
        [ 2,  5, 17],
        [ 7,  3, 12],
        [ 9, -2,  7],
        [ 5, -7,  4],
        [ 0, -8,  7],
        [-5, -5, 12],
        [-8,  0, 16],

        [-5,  6, 19],
        [ 0,  9, 16],
        [ 5,  7, 11],
        [ 8,  2,  8],
        [ 5, -3,  5],
        [ 0, -5,  9],
        [-4, -2, 14],
        [-7,  3, 18],

        [-4,  8, 14],
        [ 1,  7,  8],
        [ 6,  4,  4],
        [ 9,  0,  7],
        [ 6, -5, 12],
        [ 1, -8, 16],
        [-4, -6, 19],

        [-8, -1, 14],
        [-6,  5,  9],
        [-1,  9,  5],
        [ 4,  7,  8],
        [ 8,  3, 14],
        [ 6, -2, 19],
        [ 1, -6, 16],
        [-4, -8, 11],
        [-8, -4,  7],

        [-6,  2,  4],
        [-1,  6,  8],
        [ 4,  9, 13],
        [ 8,  6, 18],
        [ 5,  1, 16],
        [ 0, -3, 11],
        [-5, -6,  7],
        [-9, -8,  3]
    ], dtype=float)


def generate_flight_trajectory(num_points=2400):
    """Generate raw spline trajectory coordinates as in user's script."""
    control_points = get_user_control_points()
    spline, _ = splprep(control_points.T, s=0, k=3)
    t = np.linspace(0, 1, num_points)
    x, y, z = splev(t, spline)
    trajectory = np.column_stack((x, y, z))
    return trajectory


def generate_scaled_flight_trajectory(task_dt: float = 0.0002, v_cruise: float = 0.70):
    """
    Construct the physical flight trajectory scaled to the FlyBody arena:
    - Shifts XY coordinates so that Waypoint 0 starts at the natural arena origin (0, 0)
    - Maps altitude Z from [3, 19] grid units to biological flight envelope [0.38m, 0.65m]
    - Arc-length parameterizes path for smooth constant cruise velocity (v_cruise = 0.70 m/s)
    - Full 360° continuous tangent heading (no backwards flight)
    - Coordinated dynamic banking into turns: bank = -arctan(v_xy * dpsi / g)
    - Ground launch jump from origin (0, 0, 0.05m) to (0, 0, 0.38m)
    - Full closed-loop 42-point B-spline traversal
    - Terminal hover stabilization at closed loop destination
    """
    control_points = get_user_control_points()

    # Shift XY so start waypoint is at (0, 0)
    shifted_pts = control_points.copy()
    shifted_pts[:, 0] -= control_points[0, 0]
    shifted_pts[:, 1] -= control_points[0, 1]

    spline, _ = splprep(shifted_pts.T, s=0, k=3)

    # High-resolution sampling for arc-length parameterization
    u_fine = np.linspace(0, 1, 12000)
    x_raw, y_raw, z_raw = splev(u_fine, spline)

    scale_xy = 0.035
    z_min_fly = 0.38
    z_max_fly = 0.65
    scale_z = (z_max_fly - z_min_fly) / (19.0 - 3.0)

    x_scaled = x_raw * scale_xy
    y_scaled = y_raw * scale_xy
    z_scaled = z_min_fly + (z_raw - 3.0) * scale_z
    pts_fly = np.column_stack((x_scaled, y_scaled, z_scaled))

    # Compute arc length
    diffs = np.diff(pts_fly, axis=0)
    seg_lens = np.sqrt(np.sum(diffs**2, axis=1))
    cum_lens = np.concatenate(([0], np.cumsum(seg_lens)))
    total_length = cum_lens[-1]

    dur_acro = total_length / v_cruise
    n_acro = int(dur_acro / task_dt)
    s_eval = np.linspace(0, total_length, n_acro)

    # Arc-length reparameterization for constant cruise speed
    u_s = np.interp(s_eval, cum_lens, u_fine)
    x_s, y_s, z_s = splev(u_s, spline)
    pos_acro = np.column_stack((x_s * scale_xy, y_s * scale_xy, z_min_fly + (z_s - 3.0) * scale_z))

    # 1. Ground launch climb from (0, 0, 0.05m) to (0, 0, pos_acro[0, 2]) in 0.60s
    dur_climb = 0.60
    n_climb = int(dur_climb / task_dt)
    t_climb = np.linspace(0, 1, n_climb)
    s_curve = 3 * t_climb**2 - 2 * t_climb**3
    z_climb = 0.05 + (pos_acro[0, 2] - 0.05) * s_curve

    pos_climb = np.zeros((n_climb, 3))
    pos_climb[:, 2] = z_climb
    vel_climb = np.gradient(pos_climb, task_dt, axis=0)

    # 2. Terminal hover stabilization at destination in 1.00s
    dur_hover = 1.00
    n_hover = int(dur_hover / task_dt)
    pos_hover = np.tile(pos_acro[-1], (n_hover, 1))
    vel_hover = np.zeros((n_hover, 3))

    # Combine all position segments
    all_pos = np.concatenate([pos_climb, pos_acro, pos_hover], axis=0)
    all_vel = np.gradient(all_pos, task_dt, axis=0)
    n_total = len(all_pos)

    vx, vy, vz = all_vel[:, 0], all_vel[:, 1], all_vel[:, 2]
    v_xy = np.sqrt(vx**2 + vy**2)

    # Continuous 360° tangent heading
    psi = np.unwrap(np.arctan2(vy, vx))
    psi[:n_climb] = psi[n_climb]
    dpsi = np.gradient(psi, task_dt)

    # Pitch tilt and dynamic banking ('bending over into turns')
    gamma = np.arctan2(vz, np.maximum(v_xy, 1e-4))
    theta = np.deg2rad(-47.5) + 0.15 * gamma
    bank = -np.arctan(np.clip(v_xy * dpsi / 9.81, -np.tan(np.deg2rad(25)), np.tan(np.deg2rad(25))))
    bank[:n_climb] = 0.0
    bank[-n_hover:] = 0.0

    qpos = np.zeros((n_total, 7))
    qvel = np.zeros((n_total, 6))
    qpos[:, :3] = all_pos
    qvel[:, :3] = all_vel

    for i in range(n_total):
        qy = np.array([np.cos(theta[i]/2), 0, np.sin(theta[i]/2), 0])
        qx = np.array([np.cos(bank[i]/2), np.sin(bank[i]/2), 0, 0])
        qz = np.array([np.cos(psi[i]/2), 0, 0, np.sin(psi[i]/2)])
        qpos[i, 3:] = mult_quat(qz, mult_quat(qx, qy))

    total_duration = dur_climb + dur_acro + dur_hover

    # Sparse ribbon points for 3D visualization capsules (200 segments along user trajectory)
    stride = max(1, len(pos_acro) // 200)
    ribbon_points = pos_acro[::stride]

    t_climb_end = dur_climb
    t_acro_end = dur_climb + dur_acro

    phase_intervals = [
        ('1. GROUND LAUNCH & ASCENT',       0.00, t_climb_end, (0.05, pos_acro[0, 2])),
        ('2. 42-POINT ACROBATIC B-SPLINE',  t_climb_end, t_acro_end, (z_min_fly, z_max_fly)),
        ('3. TERMINAL HOVER STABILIZATION', t_acro_end, total_duration + 1.0, (pos_acro[-1, 2], pos_acro[-1, 2]))
    ]

    return qpos, qvel, total_duration, phase_intervals, bank, ribbon_points, (0.0, 0.0, 0.05)


# ==========================================================
# 2. TELEMETRY HUD OVERLAY
# ==========================================================
def add_flight_hud(frame, t, phase_name, euler_deg, pos, vel, motor_summary):
    """Render high-clarity flight telemetry HUD overlay on camera frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    overlay_h = 135
    for y in range(overlay_h):
        alpha = int(195 * (1 - y / overlay_h))
        draw.line([(0, y), (frame.shape[1], y)], fill=(0, 0, 0, alpha))

    speed_cms = np.linalg.norm(vel) * 100.0
    x_cm = pos[0] * 100.0
    y_cm = pos[1] * 100.0
    z_cm = pos[2] * 100.0

    lines = [
        f"DROSOPHILA 42-POINT 3D ACROBATIC FLIGHT  |  t={t:5.2f}s  |  True 1.0x Real-Time (60 FPS)",
        f"Phase: [{phase_name}]  |  Speed: {speed_cms:5.1f} cm/s  |  Pos: ({x_cm:+5.1f}, {y_cm:+5.1f}, {z_cm:5.1f}) cm",
        f"Attitude: Roll={euler_deg[0]:+5.1f}° (Bank)   Pitch={euler_deg[1]:+5.1f}°   Yaw={euler_deg[2]:+5.1f}°",
        f"Connectome: 10,668 Neurons (527K Synapses)  |  DLM Power: {motor_summary[0]:.0f}Hz  |  Steer: {motor_summary[1]:.0f}Hz"
    ]

    y_pos = 8
    for i, line in enumerate(lines):
        color = (255, 255, 255)
        if i == 0:
            color = (0, 255, 210)      # Bright Cyan
        elif i == 1:
            color = (255, 225, 100)    # Warm Gold
        elif i == 3:
            color = (255, 150, 90)     # Coral
        draw.text((16, y_pos), line, fill=color)
        y_pos += 26
    return np.array(img)


# ==========================================================
# 3. MAIN SIMULATION & OFFSCREEN VIDEO RENDERER
# ==========================================================
def main():
    print("=" * 75)
    print("   DROSOPHILA CONNECTOME — EXACT 42-POINT 3D ACROBATIC FLIGHT")
    print("   [10,668 Neurons | 527K Synapses | True 1.0x Biological Speed (60 FPS)]")
    print("=" * 75)

    # 1. Save trajectory coordinates to fly_trajectory.csv (user specification)
    print("[*] Generating reference trajectory from 42 control points...")
    raw_trajectory = generate_flight_trajectory(num_points=2400)
    csv_path = os.path.join(WORKSPACE_ROOT, "fly_trajectory.csv")
    np.savetxt(
        csv_path,
        raw_trajectory,
        delimiter=",",
        header="x,y,z",
        comments=""
    )
    print(f"[+] Trajectory generated: {raw_trajectory.shape}")
    print(f"[+] Saved to {csv_path}")

    # Output file paths
    output_dir = os.path.join(WORKSPACE_ROOT, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, 'acrobatic_flight_realtime.mp4')
    output_profile = os.path.join(output_dir, 'acrobatic_flight_profile.png')
    output_telemetry = os.path.join(output_dir, 'acrobatic_flight_telemetry.json')

    task_dt = 0.0002
    v_cruise = 0.70
    print(f"[*] Scaling trajectory to FlyBody physical arena at {v_cruise:.2f} m/s...")
    qpos, qvel, total_duration, phase_intervals, ref_banks, ribbon_points, start_xyz = generate_scaled_flight_trajectory(
        task_dt=task_dt, v_cruise=v_cruise
    )
    n_sim_steps = len(qpos)
    print(f"[*] Trajectory duration: {total_duration:.2f}s ({n_sim_steps} steps at 5,000 Hz)")
    print(f"[*] Start coordinate: ({start_xyz[0]:+.3f}, {start_xyz[1]:+.3f}, {start_xyz[2]:.3f}) m")
    print(f"[*] Ribbon segments for 3D visualization: {len(ribbon_points) - 1}")

    loader = InferenceFlightTrajectoryLoader()
    loader.set_next_trajectory(qpos, qvel)

    print("[*] Initializing Closed-Loop FlyBody MuJoCo + 10,668-Neuron Connectome Brain...")
    sim = ClosedLoopFlyBodySim(
        controller_type='connectome_hybrid',
        nodes_path='circuits/comprehensive_mobility_nodes.feather',
        edges_path='circuits/comprehensive_mobility_edges.feather',
        connectome_blend=0.35
    )
    env = sim.env
    env.env._task._traj_generator = loader
    env.env._task._time_limit = total_duration + 3600.0
    env.env._time_limit = total_duration + 3600.0
    sim.decoder.use_ref_tracking = False

    class AcrobaticBalancedDecoder:
        """
        Biological Motor Decoder with Haltere Gyroscopic Low-Pass Filtering
        and Neuromuscular Muscle Activation Dynamics.
        Removes intra-stroke 200 Hz wingbeat oscillation, delivering smooth acrobatic flight.
        """
        def __init__(self, sim, task_dt: float = 0.0002):
            self.sim = sim
            self.dt = task_dt
            self.alpha_gyro = task_dt / (0.008 + task_dt)
            self.alpha_vz = task_dt / (0.012 + task_dt)
            self.alpha_act = task_dt / (0.006 + task_dt)
            self.filt_ang_vel = np.zeros(3)
            self.filt_vz = 0.0
            self.prev_action = np.zeros(12)

        def reset(self):
            self.filt_ang_vel = np.zeros(3)
            self.filt_vz = 0.0
            self.prev_action = np.zeros(12)

        def decode(self, motor_rates, current_euler, state=None, target_z=None, bank_ref=0.0, t=0.0):
            r0 = self.sim.decoder.r0_steer
            r0_pwr = self.sim.decoder.r0_power
            diff_yaw = (motor_rates.get('yaw_L', r0) - r0) - (motor_rates.get('yaw_R', r0) - r0)
            diff_roll = (motor_rates.get('roll_L', r0) - r0) - (motor_rates.get('roll_R', r0) - r0)
            diff_pitch = (motor_rates.get('pitch_L', r0) - r0) - (motor_rates.get('pitch_R', r0) - r0)
            
            conn_yaw = self.sim.decoder.k_yaw * diff_yaw
            conn_roll = self.sim.decoder.k_roll * diff_roll
            conn_pitch = self.sim.decoder.k_pitch * diff_pitch
            
            dlm_mean = ((motor_rates.get('DLM_L', r0_pwr) - r0_pwr) + (motor_rates.get('DLM_R', r0_pwr) - r0_pwr)) / 2.0
            dvm_mean = ((motor_rates.get('DVM_L', r0_pwr) - r0_pwr) + (motor_rates.get('DVM_R', r0_pwr) - r0_pwr)) / 2.0
            
            # Haltere Gyroscopic Coriolis Rate Filtering
            raw_ang_vel = state.get('ang_vel', np.zeros(3)) if state else np.zeros(3)
            self.filt_ang_vel = (1.0 - self.alpha_gyro) * self.filt_ang_vel + self.alpha_gyro * raw_ang_vel
            
            # Vertical velocity filtering
            raw_vz = state.get('lin_vel', np.zeros(3))[2] if state else 0.0
            self.filt_vz = (1.0 - self.alpha_vz) * self.filt_vz + self.alpha_vz * raw_vz
            
            roll_err = current_euler[0] - bank_ref
            pitch_err = current_euler[1] - np.deg2rad(-47.5)
            
            pd_roll = -0.22 * roll_err - 0.035 * self.filt_ang_vel[0]
            pd_pitch = -0.18 * pitch_err - 0.035 * self.filt_ang_vel[1]
            pd_yaw = -0.02 * self.filt_ang_vel[2]
            
            action = np.zeros(12)
            action[3] = np.clip(conn_yaw + pd_roll, -1.0, 1.0)
            action[4] = np.clip(conn_roll + pd_pitch, -1.0, 1.0)
            action[5] = np.clip(conn_pitch, -1.0, 1.0)
            action[6] = np.clip(-conn_yaw - pd_roll, -1.0, 1.0)
            action[7] = np.clip(conn_roll + pd_pitch, -1.0, 1.0)
            action[8] = np.clip(-conn_pitch, -1.0, 1.0)
            action[9] = np.clip(0.5 * pd_yaw, -0.7, 0.7)
            
            delta_freq = self.sim.decoder.k_freq * (dlm_mean - dvm_mean)
            if target_z is not None and state is not None:
                alt_err = target_z - state['pos'][2]
                cos_roll = max(0.5, np.cos(current_euler[0]))
                bank_comp = (1.0 / cos_roll - 1.0) * 0.20
                delta_freq += 1.4 * alt_err - 0.012 * self.filt_vz + 0.16 + bank_comp
                if state['pos'][2] < 0.12 and t > 0.35:
                    delta_freq += 0.40
            action[11] = np.clip(delta_freq, -1.0, 1.0)
            
            smoothed = (1.0 - self.alpha_act) * self.prev_action + self.alpha_act * action
            self.prev_action = smoothed.copy()
            return smoothed

    decoder = AcrobaticBalancedDecoder(sim, task_dt)

    # Termination safety: terminate only on actual ground impact after takeoff ascent
    import types
    task = env.env._task
    def safe_term(task_self, physics):
        height = physics.data.qpos[2]
        t = physics.time()
        if t > 0.80 and height < 0.025:
            return True
        return False
    task.check_termination = types.MethodType(safe_term, task)

    # Camera selection: walker/track3 for elevated 3D tracking
    cam_id = 0
    for i in range(env.physics.model.ncam):
        if env.physics.model.camera(i).name == 'walker/track3':
            cam_id = i
            break
    print(f"[*] Using camera: '{env.physics.model.camera(cam_id).name}' (id={cam_id})")

    # True 1.0x Real-Time configuration: 60 FPS
    fps = 60
    render_every = int(round((1.0 / fps) / task_dt))
    print(f"[*] Real-time render configuration: {fps} FPS, render every {render_every} steps ({render_every * task_dt * 1000:.1f}ms/frame)")

    # Ground stance initialization at arena center
    ground_z = 0.05
    state = env.reset(start_on_ground=True)
    env.physics.data.qpos[:3] = [0.0, 0.0, ground_z]
    env.physics.forward()

    # Trigger middle leg jump takeoff reflex
    env.trigger_takeoff_jump(jump_speed=1.2, forward_speed=0.0)
    if sim.circuit is not None:
        sim.circuit.reset()

    # Cyan trajectory ribbon scene callback for offscreen camera rendering
    cyan_rgba = np.array([0.10, 0.95, 0.85, 0.90], dtype=np.float32)
    capsule_radius = 0.0016  # 1.6 mm thin cyan capsule ribbon
    eye9 = np.eye(3).flatten()
    zeros3 = np.zeros(3)
    ribbon_p1 = np.ascontiguousarray(ribbon_points[:-1], dtype=np.float64)
    ribbon_p2 = np.ascontiguousarray(ribbon_points[1:], dtype=np.float64)
    n_ribbon = len(ribbon_p1)

    def trajectory_scene_callback(phys, scene):
        """Inject 3D cyan trajectory ribbon directly into offscreen camera scene."""
        all_geoms = super(scene.__class__, scene).geoms
        for i in range(n_ribbon):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = all_geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                zeros3,
                zeros3,
                eye9,
                cyan_rgba
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                capsule_radius,
                ribbon_p1[i],
                ribbon_p2[i]
            )
            geom.rgba[:] = cyan_rgba
            scene.ngeom += 1

    frames = []
    telemetry = {
        'times': [],
        'pos': [],
        'ref_pos': [],
        'euler_deg': [],
        'speed_cms': [],
        'phases': [],
        'dlm_rates': [],
        'steer_rates': []
    }

    resolution = (720, 1280)
    print(f"[*] Simulating and rendering {n_sim_steps} physical steps across acrobatic trajectory...")
    t0 = time.time()

    for s in range(n_sim_steps):
        t = s * task_dt

        # Find current phase name
        phase_name = phase_intervals[-1][0]
        for z_name, z_start, z_end, _ in phase_intervals:
            if z_start <= t < z_end:
                phase_name = z_name
                break

        obs_tf = {k: tf.convert_to_tensor(np.expand_dims(v, axis=0), dtype=tf.float32) for k, v in state['raw_observation'].items()}
        base_action = sim.pretrained_policy(obs_tf).mean()[0].numpy()
        sensory = sim.encoder.update(state)
        motor_rates = sim.circuit.step(sensory)

        ref_z = qpos[s, 2]
        b_ref = ref_banks[s]
        conn_action = decoder.decode(motor_rates, state['euler'], state=state, target_z=ref_z, bank_ref=b_ref, t=t)
        action = base_action + sim.connectome_blend * conn_action
        action = np.clip(action, env.env.action_spec().minimum, env.env.action_spec().maximum)

        state, r, d = env.step(action)

        pos = state['pos'].copy()
        euler = state['euler'].copy()
        vel = state['lin_vel'].copy()
        euler_deg = euler * 180.0 / np.pi
        speed_cms = np.linalg.norm(vel) * 100.0

        dlm = motor_rates.get('DLM_L', 0)
        steer = np.mean([motor_rates.get('roll_L', 0), motor_rates.get('pitch_L', 0)])

        telemetry['times'].append(float(t))
        telemetry['pos'].append([float(x) for x in pos])
        telemetry['ref_pos'].append([float(x) for x in qpos[s, :3]])
        telemetry['euler_deg'].append([float(x) for x in euler_deg])
        telemetry['speed_cms'].append(float(speed_cms))
        telemetry['phases'].append(phase_name)
        telemetry['dlm_rates'].append(float(dlm))
        telemetry['steer_rates'].append(float(steer))

        # Render frame at 60 fps with 3D cyan trajectory ribbon
        if s % render_every == 0:
            pixels = env.physics.render(
                height=resolution[0],
                width=resolution[1],
                camera_id=cam_id,
                scene_callback=trajectory_scene_callback
            )
            motor_summary = (dlm, steer)
            frame = add_flight_hud(pixels, t, phase_name, euler_deg, pos, vel, motor_summary)
            frames.append(frame)

        if s % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  Step {s:5d}/{n_sim_steps} (t={t:4.2f}s) | [{phase_name[:25]}] Z={pos[2]*100:4.1f}cm, Speed={speed_cms:4.1f}cm/s, Frames={len(frames)} ({elapsed:.1f}s elapsed)")

        if d and s > 3000:
            print(f"[!] Flight terminated early at step {s}")
            break

    # Save True 1.0x Real-Time MP4 Video
    print(f"\n[*] Encoding {len(frames)} frames into True 1.0x Real-Time 60 FPS MP4 video: {output_video}...")
    imageio.mimsave(output_video, frames, fps=fps, quality=9)
    vid_size_mb = os.path.getsize(output_video) / (1024 * 1024)
    duration_s = len(frames) / fps
    print(f"[+] Video successfully saved: {vid_size_mb:.2f} MB (Playback Duration: {duration_s:.2f}s at {fps} FPS)")

    # Save telemetry JSON
    with open(output_telemetry, 'w') as f:
        json.dump(telemetry, f, indent=2)
    print(f"[+] Telemetry saved to {output_telemetry}")

    # Generate Publication Figures
    print(f"[*] Rendering 3D spatial map & telemetry analysis to {output_profile}...")
    times = np.array(telemetry['times'])
    pos_arr = np.array(telemetry['pos'])
    ref_arr = np.array(telemetry['ref_pos'])
    eul_arr = np.array(telemetry['euler_deg'])
    spd_arr = np.array(telemetry['speed_cms'])
    dlm_arr = np.array(telemetry['dlm_rates'])
    str_arr = np.array(telemetry['steer_rates'])

    fig = plt.figure(figsize=(18, 12))

    # 1. 3D Spatial Trajectory in Arena (matching user's reference image!)
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    p = ax1.scatter(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], c=times, cmap='cool', s=4, alpha=0.85)
    ax1.plot(ref_arr[:, 0], ref_arr[:, 1], ref_arr[:, 2], color='cyan', lw=1.5, alpha=0.6, label='B-Spline Target (User 42-pts)')
    ax1.scatter([pos_arr[0, 0]], [pos_arr[0, 1]], [pos_arr[0, 2]], color='lime', s=120, marker='^', label='Ground Launch')
    ax1.scatter([pos_arr[-1, 0]], [pos_arr[-1, 1]], [pos_arr[-1, 2]], color='red', s=140, marker='*', label='Station Hover')
    ax1.set_xlabel('X (meters)', fontsize=9, labelpad=6)
    ax1.set_ylabel('Y (meters)', fontsize=9, labelpad=6)
    ax1.set_zlabel('Altitude Z (meters)', fontsize=9, labelpad=6)
    ax1.set_title('3D Acrobatic 42-Waypoint Flight Path', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7)
    fig.colorbar(p, ax=ax1, shrink=0.6, pad=0.1, label='Time (s)')

    # 2. 2D Top-Down Arena Coverage (X vs Y)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(pos_arr[:, 0], pos_arr[:, 1], 'b-', lw=1.8, label='Actual Path')
    ax2.plot(ref_arr[:, 0], ref_arr[:, 1], 'c--', alpha=0.7, label='Reference Trajectory')
    ax2.scatter([pos_arr[0, 0]], [pos_arr[0, 1]], color='lime', s=100, marker='^')
    ax2.scatter([pos_arr[-1, 0]], [pos_arr[-1, 1]], color='red', s=100, marker='*')
    ax2.set_xlabel('Arena X (meters)', fontsize=9)
    ax2.set_ylabel('Arena Y (meters)', fontsize=9)
    ax2.set_title('Top-Down Multi-Loop Coverage', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    # 3. Altitude Z Profile & Vertical Ascent / Dives
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(times, pos_arr[:, 2] * 100, 'b-', lw=2.0, label='Actual Z')
    ax3.plot(times, ref_arr[:, 2] * 100, 'k--', alpha=0.5, label='Target Z')
    ax3.set_xlabel('Time (seconds)', fontsize=9)
    ax3.set_ylabel('Altitude (cm)', fontsize=9)
    ax3.set_title('Altitude Profile: Helical Ascents & Dives', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # 4. Flight Speed Profile
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(times, spd_arr, color='crimson', lw=1.8)
    ax4.axhline(np.mean(spd_arr), color='darkred', linestyle=':', label=f'Mean = {np.mean(spd_arr):.1f} cm/s')
    ax4.set_xlabel('Time (seconds)', fontsize=9)
    ax4.set_ylabel('Speed (cm/s)', fontsize=9)
    ax4.set_title(f'Flight Speed (Peak = {np.max(spd_arr):.1f} cm/s)', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    # 5. Dynamic Banking & Pitch Tilt ('Bending Over' Angles)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(times, eul_arr[:, 0], 'purple', lw=1.8, label='Roll Bank Angle (deg)')
    ax5.plot(times, eul_arr[:, 1], 'teal', lw=1.5, alpha=0.8, label='Pitch Tilt (deg)')
    ax5.set_xlabel('Time (seconds)', fontsize=9)
    ax5.set_ylabel('Angle (degrees)', fontsize=9)
    ax5.set_title('Attitude Dynamics: Saccadic Banking & Pitch Tilt', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=8)

    # 6. Connectome Firing Rates
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(times, dlm_arr, color='orange', lw=1.8, label='DLM Power Muscles (Hz)')
    ax6.plot(times, str_arr, color='green', lw=1.5, label='Steering Muscles (Hz)')
    ax6.set_xlabel('Time (seconds)', fontsize=9)
    ax6.set_ylabel('Firing Rate (Hz)', fontsize=9)
    ax6.set_title('Connectome Motor Output (10,668 Neurons)', fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(output_profile, dpi=120)
    import shutil
    asset_profile = os.path.join(WORKSPACE_ROOT, 'assets', 'acrobatic_flight_profile.png')
    shutil.copy(output_profile, asset_profile)
    plt.close()
    print(f"[+] Publication figure saved to {output_profile} and {asset_profile}")
    print(f"\n[✓] ALL 42-POINT ACROBATIC DELIVERABLES GENERATED in {time.time() - t0:.1f}s!")


if __name__ == '__main__':
    main()
