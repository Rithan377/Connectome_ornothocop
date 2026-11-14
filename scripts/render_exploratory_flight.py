#!/usr/bin/env python3
"""
True 1.0x Real-Time Multi-Place Free-Flight Demonstration.

Executes a seamless, omnidirectional 7-zone exploratory flight mission using the
10,668-neuron Male Drosophila Comprehensive Mobility Brain (527K synapses)
with FlyBody MuJoCo:
  1. Ground Launch & Steep Climb: (0.00, 0.00, 0.05m) -> (0.00, 0.00, 0.55m)
  2. Zone A (Forward Sprint): 80 cm/s high-speed dash across arena
  3. Zone B (Sharp Left Saccade): 50 cm/s with 2.5 rad/s high-rate bank turn
  4. Zone C (Cross-Arena Diagonal): 75 cm/s with 12° banking across the arena
  5. Zone D (Circling Landmark Loop): 45 cm/s, continuous 2.2 rad/s loop
  6. Zone E (Reverse Saccade): 70 cm/s dynamic return sprint toward origin
  7. Zone F (Precision Destination Hover): 0 cm/s stationary altitude hold

Encoded at True 1.0x Real-Time Biological Playback (60 FPS):
  1 second of video = 1 second of physical Drosophila flight!

Outputs:
  - visualizations/exploratory_free_flight_realtime.mp4
  - visualizations/exploratory_flight_3d_profile.png
  - visualizations/exploratory_flight_telemetry.json
"""

import os
import sys
import json
import time
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


def generate_exploratory_mission_trajectory(task_dt: float = 0.0002):
    """
    Construct a smooth 7-zone exploratory free-flight trajectory covering multiple
    locations across 3D arena space using chained constant_speed segments.
    """
    target_z = 0.55
    ground_z = 0.05

    # (zone_name, duration_s, speed_m_s, yaw_speed_rad_s, bank_deg, d_heading_rad)
    zones = [
        ('1. GROUND LAUNCH & ASCENT',     0.40, 0.00, 0.00,  0.0, 0.0),
        ('2. ZONE A: FORWARD SPRINT',      0.80, 0.80, 0.00,  0.0, 0.0),
        ('3. ZONE B: SHARP LEFT SACCADE',  0.80, 0.50, 2.50,  0.0, 0.0),
        ('4. ZONE C: CROSS-ARENA DIAGONAL',0.90, 0.75, 0.00, 12.0, np.deg2rad(30.0)),
        ('5. ZONE D: CIRCLING LOOP',       1.00, 0.45, -2.20, 0.0, 0.0),
        ('6. ZONE E: REVERSE SPRINT',      0.80, 0.70, 1.80,  0.0, 0.0),
        ('7. ZONE F: DESTINATION HOVER',   0.80, 0.00, 0.00,  0.0, 0.0),
    ]

    total_duration = sum(z[1] for z in zones)
    total_steps = int(np.ceil(total_duration / task_dt)) + 800

    qpos = np.zeros((total_steps, 7), dtype=np.float64)
    qvel = np.zeros((total_steps, 6), dtype=np.float64)

    # 1. Takeoff climb
    n_climb = int(0.40 / task_dt)
    t_climb = np.linspace(0, 1, n_climb)
    s_curve = 3 * t_climb**2 - 2 * t_climb**3
    climb_z = ground_z + (target_z - ground_z) * s_curve

    pitch_y = np.deg2rad(-47.5)
    base_quat = np.array([np.cos(pitch_y/2), 0.0, np.sin(pitch_y/2), 0.0])

    for i in range(n_climb):
        qpos[i, :3] = [0.0, 0.0, climb_z[i]]
        qpos[i, 3:7] = base_quat
        qvel[i, 2] = np.gradient(climb_z, task_dt)[i]

    cur_step = n_climb
    cur_pos = np.array([0.0, 0.0, target_z])
    cur_heading = 0.0

    zone_intervals = [(zones[0][0], 0.0, 0.40)]
    t_accum = 0.40

    # 2. Chain subsequent free-flight zones
    for z_name, z_dur, sp, yw, bk, hd in zones[1:]:
        n_z = int(z_dur / task_dt)
        p_qpos, p_qvel = constant_speed_trajectory(
            n_steps=n_z,
            speed=sp,
            yaw_speed=yw,
            init_pos=tuple(cur_pos),
            init_heading=cur_heading + hd,
            body_rot_angle_y=-47.5,
            body_rot_angle_x=bk,
            control_timestep=task_dt
        )
        end_idx = min(cur_step + n_z, total_steps)
        act_len = end_idx - cur_step
        qpos[cur_step:end_idx] = p_qpos[:act_len]
        qvel[cur_step:end_idx] = p_qvel[:act_len]

        cur_pos = qpos[end_idx - 1, :3]
        cur_step = end_idx
        cur_heading += yw * z_dur
        zone_intervals.append((z_name, t_accum, t_accum + z_dur))
        t_accum += z_dur

    # Pad buffer
    while cur_step < total_steps:
        qpos[cur_step] = qpos[cur_step - 1]
        qvel[cur_step] = 0.0
        cur_step += 1

    return qpos, qvel, total_duration, zone_intervals


build_exploratory_trajectory = generate_exploratory_mission_trajectory


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
        f"DROSOPHILA FREE-FLIGHT EXPLORATION  |  t={t:5.2f}s  |  True 1.0x Biological Speed (60 FPS)",
        f"Zone:  [{phase_name}]  |  Speed: {speed_cms:5.1f} cm/s  |  Arena Pos: ({x_cm:+5.1f}, {y_cm:+5.1f}, {z_cm:5.1f}) cm",
        f"Attitude:  Roll={euler_deg[0]:+5.1f}° (Bank)   Pitch={euler_deg[1]:+5.1f}°   Yaw={euler_deg[2]:+5.1f}°",
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
    print("   DROSOPHILA CONNECTOME — MULTI-PLACE EXPLORATORY FREE-FLIGHT")
    print("   [10,668 Neurons | 527K Synapses | True 1.0x Biological Speed (60 FPS)]")
    print("=" * 75)

    output_dir = os.path.join(WORKSPACE_ROOT, 'visualizations')
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, 'exploratory_free_flight_realtime.mp4')
    output_profile = os.path.join(output_dir, 'exploratory_flight_3d_profile.png')
    output_telemetry = os.path.join(output_dir, 'exploratory_flight_telemetry.json')

    task_dt = 0.0002
    print("[*] Generating 7-zone exploratory free-flight trajectory...")
    qpos, qvel, total_duration, zone_intervals = generate_exploratory_mission_trajectory(task_dt)
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

    # Tracking camera selection: walker/track2 for wide 3/4 tracking
    cam_id = 0
    for i in range(env.physics.model.ncam):
        if env.physics.model.camera(i).name == 'walker/track2':
            cam_id = i
            break
    print(f"[*] Using camera: '{env.physics.model.camera(cam_id).name}' (id={cam_id})")

    # True 1.0x Real-Time configuration: 60 FPS
    fps = 60
    render_every = int(round((1.0 / fps) / task_dt)) # 83 steps = 16.6ms
    print(f"[*] Real-time render configuration: {fps} FPS, render every {render_every} steps ({render_every * task_dt * 1000:.1f}ms/frame)")

    # Ground stance initialization
    ground_z = 0.05
    state = env.reset(start_on_ground=True)
    env.physics.data.qpos[0:3] = [0.0, 0.0, ground_z]
    theta = np.deg2rad(-47.5)
    env.physics.data.qpos[3:7] = [np.cos(theta/2), 0.0, np.sin(theta/2), 0.0]
    env.physics.data.qvel[:] = 0.0
    env.physics.forward()

    # Trigger Giant Fiber / TTM middle leg jump takeoff reflex
    env.trigger_takeoff_jump(jump_speed=1.2, forward_speed=0.0)
    if sim.circuit is not None:
        sim.circuit.reset()

    frames = []
    telemetry = {
        'times': [],
        'pos': [],
        'euler_deg': [],
        'speed_cms': [],
        'phases': [],
        'dlm_rates': [],
        'steer_rates': []
    }

    resolution = (720, 1280) # 720p HD
    print(f"[*] Simulating and rendering {n_sim_steps} physical steps across 7 zones...")
    t0 = time.time()

    for s in range(n_sim_steps):
        t = s * task_dt

        # Find current zone name
        phase_name = zone_intervals[-1][0]
        for z_name, z_start, z_end in zone_intervals:
            if z_start <= t < z_end:
                phase_name = z_name
                break

        # Observation and hybrid control
        obs_tf = {k: tf.convert_to_tensor(np.expand_dims(v, axis=0), dtype=tf.float32) for k, v in state['raw_observation'].items()}
        base_action = sim.pretrained_policy(obs_tf).mean()[0].numpy()
        sensory = sim.encoder.update(state)
        motor_rates = sim.circuit.step(sensory)

        ref_z = qpos[s, 2]
        conn_action = sim.decoder.decode(motor_rates, state['euler'], state=state, target_z=ref_z)
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

    # Generate 3D Spatial Map & Telemetry Publication Figures
    print(f"[*] Rendering 3D spatial map & telemetry analysis to {output_profile}...")
    times = np.array(telemetry['times'])
    pos_arr = np.array(telemetry['pos'])
    eul_arr = np.array(telemetry['euler_deg'])
    spd_arr = np.array(telemetry['speed_cms'])
    dlm_arr = np.array(telemetry['dlm_rates'])
    str_arr = np.array(telemetry['steer_rates'])

    fig = plt.figure(figsize=(18, 12))

    # 1. 3D Spatial Trajectory in Arena
    ax1 = fig.add_subplot(2, 3, 1, projection='3d')
    p = ax1.scatter(pos_arr[:, 0], pos_arr[:, 1], pos_arr[:, 2], c=times, cmap='viridis', s=6, alpha=0.8)
    ax1.scatter([pos_arr[0, 0]], [pos_arr[0, 1]], [pos_arr[0, 2]], color='green', s=120, marker='^', label='Ground Takeoff (0,0,5cm)')
    ax1.scatter([pos_arr[-1, 0]], [pos_arr[-1, 1]], [pos_arr[-1, 2]], color='red', s=140, marker='*', label='Hover Destination')
    ax1.set_xlabel('X (meters)', fontsize=10, labelpad=8)
    ax1.set_ylabel('Y (meters)', fontsize=10, labelpad=8)
    ax1.set_zlabel('Altitude Z (meters)', fontsize=10, labelpad=8)
    ax1.set_title('3D Multi-Place Exploratory Free-Flight Path', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    fig.colorbar(p, ax=ax1, shrink=0.6, pad=0.1, label='Time (s)')

    # 2. 2D Top-Down Arena Coverage (X vs Y)
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(pos_arr[:, 0], pos_arr[:, 1], 'b-', linewidth=2, label='Flown Path')
    ax2.plot(pos_arr[0, 0], pos_arr[0, 1], 'g^', markersize=10, label='Takeoff')
    ax2.plot(pos_arr[-1, 0], pos_arr[-1, 1], 'r*', markersize=12, label='Destination')
    ax2.set_xlabel('Arena X (meters)', fontsize=10)
    ax2.set_ylabel('Arena Y (meters)', fontsize=10)
    ax2.set_title('Top-Down Arena Coverage (X vs Y)', fontsize=12, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend(loc='lower right', fontsize=8)

    # 3. Flight Speed Profile
    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(times, spd_arr, 'forestgreen', linewidth=2, label='Speed (cm/s)')
    ax3.axhline(70.0, color='gray', linestyle=':', label='Standard Cruise (70 cm/s)')
    ax3.set_xlabel('Time (seconds)', fontsize=10)
    ax3.set_ylabel('Speed (cm/s)', fontsize=10)
    ax3.set_title('Flight Velocity Profile (True Biological Speed)', fontsize=12, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(loc='upper right', fontsize=8)

    # 4. Altitude Profile (Z vs Time)
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(times, pos_arr[:, 2] * 100.0, 'navy', linewidth=2, label='Altitude (cm)')
    ax4.axhline(5.0, color='black', linestyle='--', alpha=0.5, label='Floor (5 cm)')
    ax4.axhline(55.0, color='crimson', linestyle=':', label='Cruise Altitude (55 cm)')
    ax4.set_xlabel('Time (seconds)', fontsize=10)
    ax4.set_ylabel('Altitude (cm)', fontsize=10)
    ax4.set_title('Altitude Profile: Takeoff, Cruise & Hover', fontsize=12, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.6)
    ax4.legend(loc='lower right', fontsize=8)

    # 5. Attitude Dynamics (Roll, Pitch, Yaw)
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(times, eul_arr[:, 0], 'red', linewidth=1.5, label='Roll (Bank)')
    ax5.plot(times, eul_arr[:, 1], 'blue', linewidth=1.5, label='Pitch')
    ax5.plot(times, eul_arr[:, 2], 'purple', linewidth=1.5, label='Yaw')
    ax5.set_xlabel('Time (seconds)', fontsize=10)
    ax5.set_ylabel('Angle (degrees)', fontsize=10)
    ax5.set_title('Attitude Dynamics & Banking Turns', fontsize=12, fontweight='bold')
    ax5.grid(True, linestyle='--', alpha=0.6)
    ax5.legend(loc='upper right', fontsize=8)

    # 6. 10,668-Neuron Connectome Motor Rates
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(times, dlm_arr, 'darkorange', linewidth=1.8, label='DLM (Wing Power)')
    ax6.plot(times, str_arr, 'teal', linewidth=1.8, label='Steering Motor Neurons')
    ax6.set_xlabel('Time (seconds)', fontsize=10)
    ax6.set_ylabel('Firing Rate (Hz)', fontsize=10)
    ax6.set_title('Connectome Neuromuscular Output', fontsize=12, fontweight='bold')
    ax6.grid(True, linestyle='--', alpha=0.6)
    ax6.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_profile, dpi=200)
    plt.close()
    print(f"[+] Publication figure saved to {output_profile}")
    print(f"\n[✓] ALL EXPLORATORY FREE-FLIGHT DELIVERABLES GENERATED SUCCESSFULLY in {time.time() - t0:.1f}s!")


if __name__ == '__main__':
    main()
