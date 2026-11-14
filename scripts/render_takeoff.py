"""
Render Drosophila Ground Takeoff Video with Comprehensive Mobility Connectome.

Starts the fruit fly grounded on the floor (z = 0.05), triggers the Giant Fiber /
TTMn explosive leg extension jump reflex, ramps up WBPG wing flapping power,
and uses the 10,668-neuron Comprehensive Mobility Brain for closed-loop
sensorimotor attitude stabilization as it takes off and climbs into mid-air!

Output:
- visualizations/connectome_ground_takeoff.mp4
- visualizations/ground_takeoff_telemetry.json
- visualizations/ground_takeoff_profile.png
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

from orthocop.flybody.closed_loop import ClosedLoopFlyBodySim


def add_takeoff_overlay(frame, t, phase_name, euler_deg, pos, ang_vel, motor_rates):
    """Add flight telemetry overlay tailored for ground takeoff demonstration."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    overlay_h = 145
    for y in range(overlay_h):
        alpha = int(190 * (1 - y / overlay_h))
        draw.line([(0, y), (frame.shape[1], y)], fill=(0, 0, 0, alpha))
    
    z_cm = pos[2] * 100.0  # display in cm
    lines = [
        f"DROSOPHILA CONNECTOME GROUND TAKEOFF  |  t={t:.3f}s  |  10,668 Neurons (527K Synapses)  |  Slow-Motion Replay",
        f"Phase:     [{phase_name}]  |  Altitude: Z={pos[2]:.3f}m ({z_cm:.1f} cm)  |  Climb Vz={ang_vel[2]:+.1f}",
        f"Attitude:  Roll={euler_deg[0]:+6.1f}°   Pitch={euler_deg[1]:+6.1f}°   Yaw={euler_deg[2]:+6.1f}°",
        f"Motor:     DLM (Wing Power)={motor_rates[0]:.0f}Hz   Steer={motor_rates[1]:.0f}Hz   Leg TTM={motor_rates[2]:.0f}Hz",
    ]
    
    y_offset = 8
    for i, line in enumerate(lines):
        color = (255, 215, 0) if i == 0 else ((0, 255, 180) if i == 1 else (200, 240, 255))
        draw.text((14, y_offset), line, fill=color)
        y_offset += 24
    
    return np.array(img)


def render_ground_takeoff(
    output_path="visualizations/connectome_ground_takeoff.mp4",
    duration=0.80,
    camera_name='walker/track2',
    resolution=(720, 1280),
    fps=30
):
    print("=" * 75)
    print(" RENDERING DROSOPHILA CONNECTOME GROUND TAKEOFF VIDEO")
    print("=" * 75)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    print("[*] Initializing FlyBody and 10,668-Neuron Comprehensive Mobility Brain...")
    sim = ClosedLoopFlyBodySim(
        controller_type='connectome_hybrid',
        nodes_path='circuits/comprehensive_mobility_nodes.feather',
        edges_path='circuits/comprehensive_mobility_edges.feather',
        connectome_blend=0.35
    )
    env = sim.env

    target_z = 0.55
    env.configure_flight_mode(
        mode='hover',
        target_z=target_z,
        duration=duration,
        start_on_ground=True,
        climb_duration=0.40
    )
    sim.decoder.use_ref_tracking = False

    # Find camera ID
    cam_id = 0
    for i in range(env.physics.model.ncam):
        if env.physics.model.camera(i).name == camera_name:
            cam_id = i
            break
    print(f"[*] Using camera '{camera_name}' (id={cam_id})")

    import types
    task = env.env._task
    def persistent_check_termination(task_self, physics):
        height = physics.data.qpos[2]
        if physics.time() > 0.15 and height < 0.025:
            return True
        return False
    task.check_termination = types.MethodType(persistent_check_termination, task)

    # Reset directly to ground stance
    state = env.reset(start_on_ground=True)
    env.set_ground_stance(ground_z=0.05, pitch_deg=-47.5)
    if sim.circuit is not None:
        sim.circuit.reset()

    # Trigger Giant Fiber / TTM middle leg jump takeoff reflex (1.2 m/s biological speed)
    env.trigger_takeoff_jump(jump_speed=1.2, forward_speed=0.0)

    dt = env.dt_ctrl
    n_steps = int(round(duration / dt))
    # Capture every 6 simulation steps for smooth high-speed camera playback
    render_every = 6

    print(f"[*] Starting ground takeoff simulation ({n_steps} steps, {duration}s)...")
    print(f"    Initial ground altitude Z = {env.physics.data.qpos[2]:.4f}m")

    frames = []
    telemetry = {
        'times': [],
        'altitude_m': [],
        'roll_deg': [],
        'pitch_deg': [],
        'yaw_deg': [],
        'climb_vel': [],
        'phases': []
    }

    for step_idx in range(n_steps):
        t = step_idx * dt
        
        pos = state['pos'].copy()
        euler = state['euler'].copy()
        omega = state['ang_vel'].copy()
        euler_deg = euler * 180.0 / np.pi

        # Classify takeoff flight phase
        if t < 0.040:
            phase = "1. GROUND ESCAPE JUMP (TTMn KICK)"
        elif t < 0.400:
            phase = "2. WING POWER CLIMB & LIFT-OFF"
        else:
            phase = "3. AIRBORNE CRUISE & STABILIZATION"

        telemetry['times'].append(float(t))
        telemetry['altitude_m'].append(float(pos[2]))
        telemetry['roll_deg'].append(float(euler_deg[0]))
        telemetry['pitch_deg'].append(float(euler_deg[1]))
        telemetry['yaw_deg'].append(float(euler_deg[2]))
        telemetry['climb_vel'].append(float(env.physics.data.qvel[2]))
        telemetry['phases'].append(phase)

        # Control computation: Hybrid (policy + connectome correction)
        import tensorflow as tf
        obs_tf = {
            k: tf.convert_to_tensor(np.expand_dims(v, axis=0), dtype=tf.float32)
            for k, v in state['raw_observation'].items()
        }
        base_action = sim.pretrained_policy(obs_tf).mean()[0].numpy()
        sensory = sim.encoder.update(state)
        motor_rates = sim.circuit.step(sensory)
        conn_action = sim.decoder.decode(motor_rates, euler, state=state, target_z=target_z)

        action = base_action + sim.connectome_blend * conn_action
        action = np.clip(action, env.env.action_spec().minimum, env.env.action_spec().maximum)

        # Physics step
        state, reward, done = env.step(action)

        # Render video frame
        if step_idx % render_every == 0:
            pixels = env.physics.render(height=resolution[0], width=resolution[1], camera_id=cam_id)
            
            pwr_rate = np.mean([motor_rates.get('DLM_L', 0), motor_rates.get('DLM_R', 0)])
            steer_rate = np.mean([motor_rates.get('roll_L', 0), motor_rates.get('pitch_L', 0)])
            leg_rate = np.mean([motor_rates.get('ttm_jump_L', 0), motor_rates.get('leg_ml_L', 0)])
            motor_summary = [pwr_rate, steer_rate, leg_rate]

            frame = add_takeoff_overlay(pixels, t, phase, euler_deg, pos, omega, motor_summary)
            frames.append(frame)

        if done:
            print(f"    [!] Sequence finished at t={t:.3f}s")
            break

    print(f"[*] Encoding {len(frames)} frames to {output_path}...")
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264', quality=8, pixelformat='yuv420p')
    for f in frames:
        writer.append_data(f)
    writer.close()
    print(f"[+] Video successfully saved: {output_path}")

    # Save telemetry JSON
    json_path = "visualizations/ground_takeoff_telemetry.json"
    with open(json_path, 'w') as f_json:
        json.dump(telemetry, f_json, indent=2)
    print(f"[+] Saved telemetry: {json_path}")

    # Generate takeoff profile chart
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    
    times = telemetry['times']
    alts = np.array(telemetry['altitude_m']) * 100.0 # cm
    rolls = telemetry['roll_deg']
    pitches = telemetry['pitch_deg']

    axes[0].plot(times, alts, 'g-', linewidth=2.2, label='Altitude (cm)')
    axes[0].axhline(5.0, color='gray', linestyle=':', label='Ground Stance (5 cm)')
    axes[0].set_ylabel('Altitude (cm)')
    axes[0].set_title('Drosophila Ground Takeoff & Climb Profile (10,668-Neuron Connectome)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper left')

    axes[1].plot(times, pitches, 'b-', linewidth=1.8, label='Pitch Angle (deg)')
    axes[1].plot(times, rolls, 'm--', linewidth=1.8, label='Roll Angle (deg)')
    axes[1].axhline(45.0, color='orange', linestyle=':', label='Hover/Climb Trim (45°)')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Angle (deg)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    chart_path = "visualizations/ground_takeoff_profile.png"
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"[+] Saved takeoff profile chart: {chart_path}")
    print(f"[+] Summary: Launch Z={alts[0]:.1f} cm -> Airborne Peak Z={np.max(alts):.1f} cm (Gain: +{np.max(alts)-alts[0]:.1f} cm)")
    print("=" * 75)


if __name__ == '__main__':
    render_ground_takeoff()
