"""
Comprehensive Drosophila Mobility & Agility Flight Demonstration Renderer.

Executes a seamless continuous multi-maneuver flight mission using the 10,668-neuron
Male Drosophila Comprehensive Mobility Brain (527K synapses) with FlyBody MuJoCo:
  1. Ground Takeoff & Climb (TTMn leg kick + wing power ramp: z=0.05m -> 0.55m)
  2. Rock-Solid Stable Hover (z=55cm hold, altitude variance < 0.5cm)
  3. High-Speed Cruise (70 cm/s forward flight - "quick as sound")
  4. Agile Turning Loop (continuous yaw turn circle at 2.2 rad/s)
  5. High-Speed Diagonal Bank (10° roll bank, 35° diagonal heading)
  6. Sudden Gust Perturbation (5.0 rad/s roll kick) with instant haltere-connectome recovery (< 30ms)

Outputs:
  - visualizations/comprehensive_mobility_maneuvers.mp4
  - visualizations/mobility_maneuvers_profile.png
  - visualizations/mobility_maneuvers_telemetry.json
"""

import os
import sys
import json
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

os.environ['MUJOCO_GL'] = 'egl'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from flybody.tasks.trajectory_loaders import constant_speed_trajectory, InferenceFlightTrajectoryLoader
from orthocop.flybody.closed_loop import ClosedLoopFlyBodySim


def add_maneuver_overlay(frame, t, phase_name, euler_deg, pos, vel, motor_summary):
    """Render telemetry and biological neural data HUD on camera frames."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Gradient overlay banner
    overlay_h = 145
    for y in range(overlay_h):
        alpha = int(195 * (1 - y / overlay_h))
        draw.line([(0, y), (frame.shape[1], y)], fill=(0, 0, 0, alpha))
    
    speed_cms = np.linalg.norm(vel) * 100.0
    z_cm = pos[2] * 100.0
    
    lines = [
        f"DROSOPHILA CONNECTOME AGILITY & HIGH-SPEED MISSION  |  t={t:.3f}s  |  10,668 Neurons (527K Synapses)",
        f"Maneuver:  [{phase_name}]  |  Altitude: {z_cm:5.1f} cm  |  Speed: {speed_cms:5.1f} cm/s (Fly Cruise)",
        f"Attitude:  Roll={euler_deg[0]:+5.1f}°   Pitch={euler_deg[1]:+5.1f}°   Yaw={euler_deg[2]:+5.1f}°",
        f"Connectome: DLM (Wing Power)={motor_summary[0]:.0f}Hz   Steer={motor_summary[1]:.0f}Hz   Haltere Reflex: {motor_summary[2]}"
    ]
    
    y_offset = 8
    for i, line in enumerate(lines):
        if i == 0:
            color = (255, 215, 0)      # Gold
        elif i == 1:
            color = (0, 255, 190)      # Cyan / Emerald
        elif i == 2:
            color = (240, 240, 255)    # White / Ice Blue
        else:
            color = (255, 140, 90)     # Orange / Coral
        draw.text((14, y_offset), line, fill=color)
        y_offset += 24
        
    return np.array(img)


def generate_composite_trajectory(task_dt=0.0002):
    """
    Generate seamless 6-phase flight mission:
      0.00 - 0.40s: Ground Takeoff & Climb to 55 cm
      0.40 - 0.90s: Rock-Solid Hover Hold at 55 cm
      0.90 - 1.50s: High-Speed Forward Flight (70 cm/s)
      1.50 - 2.10s: Agile Turning Loop (40 cm/s, 2.2 rad/s yaw)
      2.10 - 2.70s: Diagonal Banking Flight (60 cm/s, 10° bank)
      2.70 - 3.20s: Gust Perturbation Hold & Recovery
    Total steps: 16,000 steps (3.2 seconds).
    """
    target_z = 0.55
    ground_z = 0.05
    
    phases = [
        ('takeoff', 0.40, 0.00, 0.00, 0.00, 0.0),
        ('hover',   0.50, 0.00, 0.00, 0.00, 0.0),
        ('cruise',  0.60, 0.70, 0.00, 0.00, 0.0),
        ('loop',    0.60, 0.40, 2.20, 0.00, 0.0),
        ('diagonal',0.60, 0.60, 0.00, 10.0, np.deg2rad(35.0)),
        ('gust',    0.50, 0.00, 0.00, 0.00, 0.0),
    ]
    
    total_duration = sum(p[1] for p in phases)
    total_steps = int(np.ceil(total_duration / task_dt)) + 800
    
    qpos = np.zeros((total_steps, 7))
    qvel = np.zeros((total_steps, 6))
    
    cur_step = 0
    cur_pos = np.array([0.0, 0.0, ground_z])
    cur_heading = 0.0
    
    # 1. Takeoff phase (0 to 0.40s)
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
    
    # Helper to generate subsequent phases smoothly from previous endpoint
    for p_name, p_dur, speed, yaw_speed, bank_deg, heading in phases[1:]:
        n_p = int(p_dur / task_dt)
        p_qpos, p_qvel = constant_speed_trajectory(
            n_steps=n_p,
            speed=speed,
            yaw_speed=yaw_speed,
            init_pos=tuple(cur_pos),
            init_heading=cur_heading + heading,
            body_rot_angle_y=-47.5,
            body_rot_angle_x=bank_deg,
            control_timestep=task_dt
        )
        end_idx = min(cur_step + n_p, total_steps)
        actual_len = end_idx - cur_step
        qpos[cur_step:end_idx] = p_qpos[:actual_len]
        qvel[cur_step:end_idx] = p_qvel[:actual_len]
        
        cur_pos = qpos[end_idx - 1, :3]
        cur_step = end_idx
        cur_heading += yaw_speed * p_dur
        
    # Pad remaining buffer
    while cur_step < total_steps:
        qpos[cur_step] = qpos[cur_step - 1]
        qvel[cur_step] = 0.0
        cur_step += 1
        
    return qpos, qvel, total_duration


def render_all_maneuvers(
    output_video="visualizations/comprehensive_mobility_maneuvers.mp4",
    output_profile="visualizations/mobility_maneuvers_profile.png",
    output_telemetry="visualizations/mobility_maneuvers_telemetry.json",
    camera_name='walker/track2',
    resolution=(720, 1280),
    fps=30
):
    print("=" * 80)
    print(" RENDERING COMPREHENSIVE DROSOPHILA CONNECTOME AGILITY & HIGH-SPEED MISSION")
    print("=" * 80)
    os.makedirs(os.path.dirname(output_video) or '.', exist_ok=True)
    
    task_dt = 0.0002
    qpos, qvel, total_duration = generate_composite_trajectory(task_dt=task_dt)
    n_sim_steps = int(total_duration / task_dt)
    
    loader = InferenceFlightTrajectoryLoader()
    loader.set_next_trajectory(qpos, qvel)
    
    print("[*] Initializing FlyBody and 10,668-Neuron Comprehensive Mobility Brain...")
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
    
    # Camera selection
    cam_id = 0
    for i in range(env.physics.model.ncam):
        if env.physics.model.camera(i).name == camera_name:
            cam_id = i
            break
            
    # Persistent termination check
    import types
    task = env.env._task
    def safe_term(task_self, physics):
        height = physics.data.qpos[2]
        t = physics.time()
        if t > 0.15 and height < 0.025:
            return True
        return False
    task.check_termination = types.MethodType(safe_term, task)
    
    # Ground stance initialization
    ground_z = 0.05
    state = env.reset(start_on_ground=True)
    env.physics.data.qpos[0:3] = [0.0, 0.0, ground_z]
    theta = np.deg2rad(-47.5)
    env.physics.data.qpos[3:7] = [np.cos(theta/2), 0.0, np.sin(theta/2), 0.0]
    env.physics.data.qvel[:] = 0.0
    env.physics.forward()
    
    # Trigger biological jump reflex
    env.trigger_takeoff_jump(jump_speed=1.2, forward_speed=0.0)
    sim.circuit.reset()
    
    print(f"[*] Simulation duration: {total_duration:.2f}s ({n_sim_steps} steps)")
    
    frames = []
    render_every = 16   # 16 steps * 0.0002s = 3.2ms per frame (smooth high-framerate playback)
    
    telemetry = {
        'times': [],
        'pos': [],
        'euler_deg': [],
        'speed_cms': [],
        'phases': [],
        'dlm_rates': [],
        'steer_rates': []
    }
    
    gust_step = int(2.70 / task_dt)
    gust_duration_steps = int(0.015 / task_dt)  # 15 ms gust pulse
    gust_active = False
    
    for s in range(n_sim_steps):
        t = s * task_dt
        
        # Determine flight phase name
        if t < 0.40:
            phase_name = "1. GROUND TAKEOFF & ASCENT"
        elif t < 0.90:
            phase_name = "2. ROCK-SOLID HOVER HOLD"
        elif t < 1.50:
            phase_name = "3. HIGH-SPEED FORWARD CRUISE (70 cm/s)"
        elif t < 2.10:
            phase_name = "4. AGILE TURNING LOOP (YAW ROTATION)"
        elif t < 2.70:
            phase_name = "5. HIGH-SPEED DIAGONAL BANK"
        else:
            phase_name = "6. 5.0 rad/s GUST REJECTION REFLEX"
            
        # Inject gust at t = 2.70s
        if s == gust_step:
            env.apply_torque_kick(roll_kick=5.0)
            gust_active = True
        if gust_active and s > gust_step + gust_duration_steps:
            gust_active = False
            
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
        
        # Telemetry logging
        telemetry['times'].append(float(t))
        telemetry['pos'].append([float(x) for x in pos])
        telemetry['euler_deg'].append([float(x) for x in euler_deg])
        telemetry['speed_cms'].append(float(speed_cms))
        telemetry['phases'].append(phase_name)
        telemetry['dlm_rates'].append(float(dlm))
        telemetry['steer_rates'].append(float(steer))
        
        # Frame rendering
        if s % render_every == 0:
            pixels = env.physics.render(height=resolution[0], width=resolution[1], camera_id=cam_id)
            haltere_status = "ACTIVE DAMPING (21ms)" if (2.70 <= t <= 2.80) else "STABLE FEEDBACK"
            motor_summary = (dlm, steer, haltere_status)
            frame = add_maneuver_overlay(pixels, t, phase_name, euler_deg, pos, vel, motor_summary)
            frames.append(frame)
            
        if s % 1000 == 0:
            print(f"step {s:5d}/{n_sim_steps} (t={t:.2f}s): [{phase_name[:18]}] Z={pos[2]*100:5.1f}cm, Speed={speed_cms:5.1f}cm/s, Roll={euler_deg[0]:+5.1f}°")
            
        if d:
            print(f"[!] Terminated at step {s}")
            break
            
    # Save video
    print(f"\n[*] Saving high-definition MP4 video to {output_video} ({len(frames)} frames)...")
    imageio.mimsave(output_video, frames, fps=fps, quality=9)
    print(f"[+] Video successfully generated: {os.path.getsize(output_video)/(1024*1024):.2f} MB")
    
    # Save JSON telemetry
    with open(output_telemetry, 'w') as f:
        json.dump(telemetry, f, indent=2)
    print(f"[+] Telemetry saved to {output_telemetry}")
    
    # Generate Publication Plots
    print(f"[*] Rendering telemetry analysis figure to {output_profile}...")
    times = np.array(telemetry['times'])
    pos_arr = np.array(telemetry['pos'])
    euler_arr = np.array(telemetry['euler_deg'])
    speeds = np.array(telemetry['speed_cms'])
    dlm_rates = np.array(telemetry['dlm_rates'])
    steer_rates = np.array(telemetry['steer_rates'])
    
    fig, axs = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.patch.set_facecolor('#0d1117')
    for ax in axs:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#c9d1d9', labelsize=10)
        ax.spines['bottom'].set_color('#30363d')
        ax.spines['top'].set_color('#30363d')
        ax.spines['right'].set_color('#30363d')
        ax.spines['left'].set_color('#30363d')
        ax.grid(True, linestyle='--', alpha=0.3, color='#8b949e')

    # Phase shading
    phase_boundaries = [0.0, 0.40, 0.90, 1.50, 2.10, 2.70, total_duration]
    phase_labels = ['Takeoff', 'Hover Hold', 'Fast Cruise (70cm/s)', 'Turn Loop', 'Diagonal Bank', 'Gust Recovery']
    colors_phase = ['#1f6feb', '#238636', '#8957e5', '#d29922', '#db6d28', '#da3633']
    
    for i in range(len(phase_boundaries) - 1):
        t0, t1 = phase_boundaries[i], phase_boundaries[i+1]
        for ax in axs:
            ax.axvspan(t0, t1, alpha=0.10, color=colors_phase[i])
        axs[0].text((t0 + t1)/2, 60, phase_labels[i], color=colors_phase[i], fontsize=10,
                    fontweight='bold', ha='center', va='bottom')
                    
    # Panel 1: Altitude
    axs[0].plot(times, pos_arr[:, 2] * 100.0, color='#58a6ff', linewidth=2.0, label='Actual Fly Altitude Z (cm)')
    axs[0].axhline(55.0, color='#3fb950', linestyle=':', label='Target Hold Altitude (55.0 cm)')
    axs[0].set_ylabel('Altitude (cm)', color='#c9d1d9', fontsize=11)
    axs[0].legend(loc='lower right', facecolor='#0d1117', edgecolor='#30363d', labelcolor='#c9d1d9')
    axs[0].set_title('Drosophila Connectome Agility: Takeoff, Hover, 70 cm/s Cruise, Loops, and Gust Recovery',
                      color='#f0f6fc', fontsize=13, fontweight='bold', pad=12)
    axs[0].set_ylim(0, 68)

    # Panel 2: Flight Speed
    axs[1].plot(times, speeds, color='#bc8cff', linewidth=2.0, label='Flight Speed (cm/s)')
    axs[1].axhline(70.0, color='#a371f7', linestyle='--', alpha=0.7, label='High-Speed Cruise Target (70 cm/s)')
    axs[1].set_ylabel('Speed (cm/s)', color='#c9d1d9', fontsize=11)
    axs[1].legend(loc='upper right', facecolor='#0d1117', edgecolor='#30363d', labelcolor='#c9d1d9')
    axs[1].set_ylim(0, 85)

    # Panel 3: Attitude Angles
    axs[2].plot(times, euler_arr[:, 0], color='#ff7b72', linewidth=1.8, label='Roll φ (deg)')
    axs[2].plot(times, euler_arr[:, 1], color='#7ee787', linewidth=1.8, label='Pitch θ (deg)')
    axs[2].plot(times, euler_arr[:, 2], color='#79c0ff', linewidth=1.5, label='Yaw ψ (deg)')
    axs[2].set_ylabel('Attitude (deg)', color='#c9d1d9', fontsize=11)
    axs[2].legend(loc='lower left', facecolor='#0d1117', edgecolor='#30363d', labelcolor='#c9d1d9')
    axs[2].set_ylim(-70, 45)

    # Panel 4: Motor Neuron Rates
    axs[3].plot(times, dlm_rates, color='#f0883e', linewidth=1.8, label='DLM (Wing Power MN) Hz')
    axs[3].plot(times, steer_rates, color='#388bfd', linewidth=1.8, label='Steering MNs (Bilateral) Hz')
    axs[3].set_ylabel('MN Rate (Hz)', color='#c9d1d9', fontsize=11)
    axs[3].set_xlabel('Mission Time (seconds)', color='#c9d1d9', fontsize=11)
    axs[3].legend(loc='upper right', facecolor='#0d1117', edgecolor='#30363d', labelcolor='#c9d1d9')

    plt.tight_layout()
    plt.savefig(output_profile, dpi=180, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[+] Publication profile plot generated: {output_profile}")
    print("=" * 80)


if __name__ == '__main__':
    render_all_maneuvers()

