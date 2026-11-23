"""
Acrobatic Multi-Loop Flight Simulation & Trajectory Renderer.
Matches the user's reference trajectory:
- Vertical climbing helical loops / corkscrews
- High-altitude sweeping loop with heavy banking ('bending over')
- Diagonal descent dive across arena
- Low-altitude turnaround loop near floor grid
- Diagonal climb and interlocking figure-8 loop
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

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

os.environ['MUJOCO_GL'] = 'egl'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from flybody.tasks.trajectory_loaders import constant_speed_trajectory, InferenceFlightTrajectoryLoader
from orthocop.flybody.closed_loop import ClosedLoopFlyBodySim


def mult_quat(q1, q2):
    res = np.zeros(4)
    import mujoco
    mujoco.mju_mulQuat(res, q1, q2)
    return res


def generate_acrobatic_multiloop_trajectory(task_dt: float = 0.0002):
    """
    Construct a smooth C2 continuous 3D acrobatic flight trajectory matching the
    user's reference image (helical climbs, high-altitude apex loop with banking,
    diagonal descent dive, low-altitude figure-8 turnaround loop, and diagonal climb).
    Calibrated with biological Drosophila body saccades, 360° continuous heading,
    coordinated dynamic banking, and safe altitude envelope (0.38m - 0.65m).
    """
    climb_dur = 0.60
    n_climb = int(climb_dur / task_dt)

    # 1. Takeoff climb: 0.05m -> 0.48m
    t_climb = np.linspace(0, 1, n_climb)
    s_curve = 3 * t_climb**2 - 2 * t_climb**3
    climb_z = 0.05 + (0.48 - 0.05) * s_curve
    pitch_y = np.deg2rad(-47.5)
    base_quat = np.array([np.cos(pitch_y/2), 0.0, np.sin(pitch_y/2), 0.0])

    qpos_climb = np.zeros((n_climb, 7))
    qvel_climb = np.zeros((n_climb, 6))
    for i in range(n_climb):
        qpos_climb[i, :3] = [0.0, 0.0, climb_z[i]]
        qpos_climb[i, 3:] = base_quat
        qvel_climb[i, 2] = np.gradient(climb_z, task_dt)[i]

    # 2. Acrobatic 3D Waypoints matching user trajectory image
    pts = np.array([
        [ 0.00,  0.00, 0.48], # 0: start at hover
        [ 0.25,  0.10, 0.49], # 1: forward entry
        [ 0.45,  0.35, 0.51], # 2: enter lower loop
        [ 0.25,  0.65, 0.54], # 3: lower loop arc
        [-0.15,  0.50, 0.58], # 4: ascending corkscrew
        [-0.35,  0.15, 0.62], # 5: climbing spiral
        [-0.10, -0.25, 0.64], # 6: reaching apex
        [ 0.25, -0.25, 0.65], # 7: apex sweeping turn ('bending over')
        [ 0.45,  0.05, 0.63], # 8: apex loop exit
        [ 0.15,  0.25, 0.52], # 9: enter diagonal dive
        [-0.15,  0.10, 0.42], # 10: diagonal dive through center
        [-0.30, -0.15, 0.38], # 11: low-altitude loop entry
        [-0.10, -0.38, 0.38], # 12: low loop turnaround
        [ 0.20, -0.15, 0.40], # 13: low loop exit
        [ 0.35,  0.15, 0.46], # 14: diagonal climb
        [ 0.10,  0.35, 0.52], # 15: upper figure-8 curve
        [-0.15,  0.20, 0.52], # 16: upper return
        [ 0.00,  0.00, 0.50], # 17: final hover station
    ])

    from scipy.interpolate import CubicSpline
    diffs = np.diff(pts, axis=0)
    dists = np.sqrt(np.sum(diffs**2, axis=1))
    cum_dist = np.concatenate(([0], np.cumsum(dists)))
    total_len = cum_dist[-1]

    acro_dur = 8.50
    n_acro = int(acro_dur / task_dt)

    t_acro = np.linspace(0, total_len, n_acro)
    cs = CubicSpline(cum_dist, pts, bc_type='natural')
    pos_acro = cs(t_acro)
    vel_acro = cs(t_acro, 1) * (total_len / acro_dur)

    qpos_acro = np.zeros((n_acro, 7))
    qvel_acro = np.zeros((n_acro, 6))

    qpos_acro[:, :3] = pos_acro
    qvel_acro[:, :3] = vel_acro

    vx, vy, vz = vel_acro[:, 0], vel_acro[:, 1], vel_acro[:, 2]
    v_xy = np.sqrt(vx**2 + vy**2)

    # Full 360° continuous tangent heading (eliminates backwards flight)
    psi = np.unwrap(np.arctan2(vy, vx))
    dpsi = np.gradient(psi, task_dt)

    gamma = np.arctan2(vz, np.maximum(v_xy, 1e-4))
    theta = np.deg2rad(-47.5) + 0.15 * gamma
    # Coordinated dynamic banking: roll into turns ('bending over')
    bank = -np.arctan(np.clip(v_xy * dpsi / 9.81, -np.tan(np.deg2rad(22)), np.tan(np.deg2rad(22))))

    for i in range(n_acro):
        qy = np.array([np.cos(theta[i]/2), 0, np.sin(theta[i]/2), 0])
        qx = np.array([np.cos(bank[i]/2), np.sin(bank[i]/2), 0, 0])
        qz = np.array([np.cos(psi[i]/2), 0, 0, np.sin(psi[i]/2)])
        qpos_acro[i, 3:] = mult_quat(qz, mult_quat(qx, qy))

    # Combine climb + acrobatic
    qpos = np.concatenate([qpos_climb, qpos_acro], axis=0)
    qvel = np.concatenate([qvel_climb, qvel_acro], axis=0)

    # Pad remaining buffer
    pad_steps = 1000
    qpos_pad = np.tile(qpos[-1], (pad_steps, 1))
    qvel_pad = np.zeros((pad_steps, 6))
    qpos = np.concatenate([qpos, qpos_pad], axis=0)
    qvel = np.concatenate([qvel, qvel_pad], axis=0)

    ref_banks = np.concatenate([np.zeros(n_climb), bank, np.zeros(pad_steps)])
    total_duration = climb_dur + acro_dur

    zone_intervals = [
        ('1. GROUND LAUNCH & ASCENT',       0.00, 0.60, (0.05, 0.48)),
        ('2. LOWER LOOP ENTRY & ARC',       0.60, 1.80, (0.48, 0.54)),
        ('3. HELICAL CORKSCREW ASCENT',     1.80, 3.60, (0.54, 0.64)),
        ('4. APEX SWEEPING LOOP (HEAVY BANK)', 3.60, 5.20, (0.64, 0.65)),
        ('5. DIAGONAL DESCENT DIVE',        5.20, 6.50, (0.65, 0.38)),
        ('6. LOW-ALTITUDE TURNAROUND LOOP', 6.50, 7.60, (0.38, 0.40)),
        ('7. DIAGONAL CLIMB & FIGURE-8',    7.60, 8.50, (0.40, 0.52)),
        ('8. TERMINAL DESTINATION HOVER',   8.50, total_duration, (0.52, 0.50))
    ]

    return qpos, qvel, total_duration, zone_intervals, ref_banks


def add_flight_hud(frame, t, phase_name, euler_deg, pos, vel, motor_summary):
    """Render telemetry HUD overlay on camera frame."""
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
        f"DROSOPHILA 3D ACROBATIC FLIGHT  |  t={t:5.2f}s  |  True 1.0x Real-Time (60 FPS)",
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


def main():
    print("=" * 75)
    print("   DROSOPHILA CONNECTOME — 3D ACROBATIC MULTI-LOOP FLIGHT")
    print("   [10,668 Neurons | 527K Synapses | True 1.0x Biological Speed (60 FPS)]")
    print("=" * 75)

    output_dir = os.path.join(WORKSPACE_ROOT, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, 'acrobatic_flight_realtime.mp4')
    output_profile = os.path.join(output_dir, 'acrobatic_flight_profile.png')
    output_telemetry = os.path.join(output_dir, 'acrobatic_flight_telemetry.json')

    task_dt = 0.0002
    print("[*] Generating 3D acrobatic multi-loop trajectory...")
    qpos, qvel, total_duration, zone_intervals, ref_banks = generate_acrobatic_multiloop_trajectory(task_dt)
    n_sim_steps = int(total_duration / task_dt)
    print(f"[*] Trajectory duration: {total_duration:.2f}s ({n_sim_steps} steps at 5,000 Hz)")

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

    def acrobatic_balanced_decode(motor_rates, current_euler, state=None, target_z=None, bank_ref=0.0, t=0.0):
        r0 = sim.decoder.r0_steer
        r0_pwr = sim.decoder.r0_power
        diff_yaw = (motor_rates.get('yaw_L', r0) - r0) - (motor_rates.get('yaw_R', r0) - r0)
        diff_roll = (motor_rates.get('roll_L', r0) - r0) - (motor_rates.get('roll_R', r0) - r0)
        diff_pitch = (motor_rates.get('pitch_L', r0) - r0) - (motor_rates.get('pitch_R', r0) - r0)
        
        conn_yaw = sim.decoder.k_yaw * diff_yaw
        conn_roll = sim.decoder.k_roll * diff_roll
        conn_pitch = sim.decoder.k_pitch * diff_pitch
        
        dlm_mean = ((motor_rates.get('DLM_L', r0_pwr) - r0_pwr) + (motor_rates.get('DLM_R', r0_pwr) - r0_pwr)) / 2.0
        dvm_mean = ((motor_rates.get('DVM_L', r0_pwr) - r0_pwr) + (motor_rates.get('DVM_R', r0_pwr) - r0_pwr)) / 2.0
        
        roll_err = current_euler[0] - bank_ref
        pitch_err = current_euler[1] - np.deg2rad(-47.5)
        
        ang_vel = state.get('ang_vel', np.zeros(3)) if state else np.zeros(3)
        pd_roll = -0.35 * roll_err - 0.03 * ang_vel[0]
        pd_pitch = -0.40 * pitch_err - 0.03 * ang_vel[1]
        pd_yaw = -0.02 * ang_vel[2]
        
        action = np.zeros(12)
        action[3] = np.clip(conn_yaw + pd_roll, -1.0, 1.0)
        action[4] = np.clip(conn_roll + pd_pitch, -1.0, 1.0)
        action[5] = np.clip(conn_pitch, -1.0, 1.0)
        action[6] = np.clip(-conn_yaw - pd_roll, -1.0, 1.0)
        action[7] = np.clip(conn_roll + pd_pitch, -1.0, 1.0)
        action[8] = np.clip(-conn_pitch, -1.0, 1.0)
        action[9] = np.clip(0.5 * pd_yaw, -0.7, 0.7)
        
        delta_freq = sim.decoder.k_freq * (dlm_mean - dvm_mean)
        if target_z is not None and state is not None:
            alt_err = target_z - state['pos'][2]
            vz = state['lin_vel'][2]
            cos_roll = max(0.5, np.cos(current_euler[0]))
            bank_comp = (1.0 / cos_roll - 1.0) * 0.20
            delta_freq += 1.5 * alt_err - 0.012 * vz + 0.16 + bank_comp
            # Ventral optic flow ground avoidance reflex (emergency pull-up below 12cm)
            if state['pos'][2] < 0.12 and t > 0.35:
                delta_freq += 0.40
        action[11] = np.clip(delta_freq, -1.0, 1.0)
        return action

    # Persistent safe termination: only terminate on floor crash
    import types
    task = env.env._task
    def safe_term(task_self, physics):
        height = physics.data.qpos[2]
        t = physics.time()
        if t > 0.15 and height < 0.025:
            return True
        return False
    task.check_termination = types.MethodType(safe_term, task)

    # Tracking camera selection: walker/track3 for elevated 3D tracking
    cam_id = 0
    for i in range(env.physics.model.ncam):
        if env.physics.model.camera(i).name == 'walker/track3':
            cam_id = i
            break
    print(f"[*] Using camera: '{env.physics.model.camera(cam_id).name}' (id={cam_id})")

    # True 1.0x Real-Time configuration: 60 FPS
    fps = 60
    render_every = int(round((1.0 / fps) / task_dt))  # 83 steps = 16.6ms
    print(f"[*] Real-time render configuration: {fps} FPS, render every {render_every} steps ({render_every * task_dt * 1000:.1f}ms/frame)")

    # Ground stance initialization
    ground_z = 0.05
    state = env.reset(start_on_ground=True)
    env.physics.data.qpos[:3] = [0.0, 0.0, ground_z]

    # Trigger Giant Fiber / TTM middle leg jump takeoff reflex
    env.trigger_takeoff_jump(jump_speed=1.2, forward_speed=0.0)
    if sim.circuit is not None:
        sim.circuit.reset()

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

    resolution = (720, 1280)  # 720p HD
    print(f"[*] Simulating and rendering {n_sim_steps} physical steps across 8 acrobatic phases...")
    t0 = time.time()

    for s in range(n_sim_steps):
        t = s * task_dt

        # Find current zone name
        phase_name = zone_intervals[-1][0]
        for z_name, z_start, z_end, _ in zone_intervals:
            if z_start <= t < z_end:
                phase_name = z_name
                break

        obs_tf = {k: tf.convert_to_tensor(np.expand_dims(v, axis=0), dtype=tf.float32) for k, v in state['raw_observation'].items()}
        base_action = sim.pretrained_policy(obs_tf).mean()[0].numpy()
        sensory = sim.encoder.update(state)
        motor_rates = sim.circuit.step(sensory)

        ref_z = qpos[s, 2]
        b_ref = ref_banks[s]
        conn_action = acrobatic_balanced_decode(motor_rates, state['euler'], state=state, target_z=ref_z, bank_ref=b_ref, t=t)
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

        # Render frame at 60 fps
        if s % render_every == 0:
            pixels = env.physics.render(height=resolution[0], width=resolution[1], camera_id=cam_id)
            motor_summary = (dlm, steer)
            frame = add_flight_hud(pixels, t, phase_name, euler_deg, pos, vel, motor_summary)
            frames.append(frame)

        if s % 2500 == 0:
            elapsed = time.time() - t0
            print(f"  Step {s:5d}/{n_sim_steps} (t={t:4.2f}s) | [{phase_name[:22]}] Z={pos[2]*100:4.1f}cm, Speed={speed_cms:4.1f}cm/s, Frames={len(frames)} ({elapsed:.1f}s elapsed)")

        if d and s > 2000:
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
    p = ax1.scatter(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], c=times, cmap='viridis', s=6, alpha=0.8)
    ax1.plot(ref_arr[:, 0], ref_arr[:, 1], ref_arr[:, 2], 'w--', alpha=0.4, label='Target Trajectory')
    ax1.scatter([pos_arr[0, 0]], [pos_arr[0, 1]], [pos_arr[0, 2]], color='lime', s=120, marker='^', label='Ground Launch')
    ax1.scatter([pos_arr[-1, 0]], [pos_arr[-1, 1]], [pos_arr[-1, 2]], color='red', s=140, marker='*', label='Station Hover')
    ax1.set_xlabel('X (meters)', fontsize=9, labelpad=6)
    ax1.set_ylabel('Y (meters)', fontsize=9, labelpad=6)
    ax1.set_zlabel('Altitude Z (meters)', fontsize=9, labelpad=6)
    ax1.set_title('3D Acrobatic Multi-Loop Flight Path', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=7)
    fig.colorbar(p, ax=ax1, shrink=0.6, pad=0.1, label='Time (s)')

    # 2. 2D Top-Down Arena Coverage (X vs Y)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(pos_arr[:, 0], pos_arr[:, 1], 'b-', lw=1.8, label='Actual Path')
    ax2.plot(ref_arr[:, 0], ref_arr[:, 1], 'k--', alpha=0.4, label='Reference')
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
    ax3.set_title('Altitude Profile: Helical Climb & Dive', fontsize=11, fontweight='bold')
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

    # 5. Dynamic Banking & Pitch ('Bending Over' Angles)
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
    print(f"\n[✓] ALL ACROBATIC MULTI-LOOP DELIVERABLES GENERATED SUCCESSFULLY in {time.time() - t0:.1f}s!")


if __name__ == '__main__':
    main()
