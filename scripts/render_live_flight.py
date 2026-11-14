"""
Render Connectome-Driven Flight Simulation as Video.

Runs the 897-neuron stabilized spiking circuit in closed-loop with FlyBody
MuJoCo physics and records the result as an MP4 video with telemetry overlay.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ['MUJOCO_GL'] = 'egl'

import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont

from orthocop.flybody.flybody_env import FlyBodyFlightEnv
from orthocop.flybody.connectome_brian2 import ConnectomeFlightCircuit
from orthocop.flybody.sensor_encoder import BiophysicalSensorEncoder
from orthocop.flybody.motor_decoder import MotorDecoder


def add_telemetry_overlay(frame, t, euler_deg, pos, ang_vel, motor_rates_summary, neuron_count=10668, synapse_count="527K"):
    """Add flight telemetry text overlay to rendered frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    
    # Semi-transparent overlay background
    overlay_h = 135
    for y in range(overlay_h):
        alpha = int(180 * (1 - y / overlay_h))
        draw.line([(0, y), (frame.shape[1], y)], fill=(0, 0, 0, alpha))
    
    lines = [
        f"COMPREHENSIVE MOBILITY BRAIN | t={t:.3f}s | {neuron_count} LIF neurons | {synapse_count} synapses",
        f"Attitude:  Roll={euler_deg[0]:+6.1f}°  Pitch={euler_deg[1]:+6.1f}°  Yaw={euler_deg[2]:+6.1f}°",
        f"Position:  X={pos[0]:+.3f}m  Y={pos[1]:+.3f}m  Z={pos[2]:.3f}m",
        f"AngVel:    P={ang_vel[0]:+5.1f}  Q={ang_vel[1]:+5.1f}  R={ang_vel[2]:+5.1f}  rad/s",
        f"Motor:     DLM={motor_rates_summary[0]:.0f}Hz  Steer={motor_rates_summary[1]:.0f}Hz  Legs={motor_rates_summary[2]:.0f}Hz",
    ]
    
    y_offset = 8
    for line in lines:
        draw.text((12, y_offset), line, fill=(0, 255, 128))
        y_offset += 22
    
    return np.array(img)


def render_flight_video(
    output_path="visualizations/comprehensive_mobility_flight.mp4",
    nodes_path="circuits/comprehensive_mobility_nodes.feather",
    edges_path="circuits/comprehensive_mobility_edges.feather",
    duration=1.5,
    camera_name='walker/track2',
    resolution=(720, 1280),
    fps=30,
    disturbance_type='roll_kick',
    disturbance_time=0.5,
    disturbance_magnitude=6.0,
):
    """
    Run closed-loop connectome flight and render video.
    
    Args:
        output_path: Output MP4 path.
        nodes_path: Path to connectome feather nodes.
        edges_path: Path to connectome feather edges.
        duration: Simulation duration in seconds.
        camera_name: MuJoCo camera ('walker/track2', 'walker/hero', 'walker/side').
        resolution: (height, width).
        fps: Video frame rate.
        disturbance_type: 'none', 'roll_kick', 'pitch_kick', 'yaw_kick'.
        disturbance_time: When to inject disturbance (seconds).
        disturbance_magnitude: Angular kick in rad/s.
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    print("[*] Initializing FlyBody MuJoCo environment...")
    env = FlyBodyFlightEnv()
    
    print(f"[*] Loading spiking connectome from {nodes_path}...")
    circuit = ConnectomeFlightCircuit(nodes_path=nodes_path, edges_path=edges_path)
    encoder = BiophysicalSensorEncoder()
    decoder = MotorDecoder()
    
    # Find camera ID
    cam_id = None
    for i in range(env.physics.model.ncam):
        if env.physics.model.camera(i).name == camera_name:
            cam_id = i
            break
    if cam_id is None:
        print(f"  Camera '{camera_name}' not found, using camera 0")
        cam_id = 0
    
    state = env.reset()
    circuit.reset()
    
    dt = env.dt_ctrl
    n_steps = int(round(duration / dt))
    render_every = max(1, int(round(1.0 / (fps * dt))))
    
    print(f"[*] Running {n_steps} steps ({duration}s), rendering every {render_every} steps...")
    print(f"    Disturbance: {disturbance_type} at t={disturbance_time}s ({disturbance_magnitude} rad/s)")
    
    frames = []
    disturbance_applied = False
    
    for step_idx in range(n_steps):
        t = step_idx * dt
        
        # Disturbance injection
        if not disturbance_applied and t >= disturbance_time:
            if disturbance_type == 'roll_kick':
                env.apply_torque_kick(roll_kick=disturbance_magnitude)
            elif disturbance_type == 'pitch_kick':
                env.apply_torque_kick(pitch_kick=disturbance_magnitude)
            elif disturbance_type == 'yaw_kick':
                env.apply_torque_kick(yaw_kick=disturbance_magnitude)
            disturbance_applied = True
            print(f"    [!] Disturbance applied at t={t:.3f}s")
        
        # Connectome sensory-motor loop
        sensory = encoder.update(state)
        motor_rates = circuit.step(sensory)
        action = decoder.decode(motor_rates, state['euler'])
        
        # Physics step
        state, reward, done = env.step(action)
        
        # Render frame
        if step_idx % render_every == 0:
            pixels = env.physics.render(
                height=resolution[0], width=resolution[1], camera_id=cam_id
            )
            
            euler_deg = state['euler'] * 180.0 / np.pi
            power_rate = np.mean([motor_rates.get('DLM_L', 0), motor_rates.get('DLM_R', 0),
                                  motor_rates.get('DVM_L', 0), motor_rates.get('DVM_R', 0)])
            steer_rate = np.mean([motor_rates.get('roll_L', 0), motor_rates.get('roll_R', 0),
                                  motor_rates.get('pitch_L', 0), motor_rates.get('pitch_R', 0),
                                  motor_rates.get('yaw_L', 0), motor_rates.get('yaw_R', 0)])
            leg_rate = np.mean([motor_rates.get('leg_fl_L', 30), motor_rates.get('leg_fl_R', 30),
                                motor_rates.get('leg_ml_L', 30), motor_rates.get('leg_ml_R', 30),
                                motor_rates.get('leg_hl_L', 30), motor_rates.get('leg_hl_R', 30)])
            motor_summary = [power_rate, steer_rate, leg_rate]
            
            frame = add_telemetry_overlay(
                pixels, t, euler_deg, state['pos'], state['ang_vel'], motor_summary,
                neuron_count=circuit.num_neurons, synapse_count=f"{len(circuit.edges_df)//1000}K"
            )
            frames.append(frame)
        
        if done:
            print(f"    [x] Episode terminated at t={t:.3f}s")
            break
    
    print(f"[*] Encoding {len(frames)} frames to {output_path}...")
    writer = imageio.get_writer(output_path, fps=fps, codec='libx264',
                                 quality=8, pixelformat='yuv420p')
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    
    euler_final = state['euler'] * 180.0 / np.pi
    print(f"[+] Video saved: {output_path}")
    print(f"    Final attitude: Roll={euler_final[0]:+.1f}° Pitch={euler_final[1]:+.1f}° Yaw={euler_final[2]:+.1f}°")
    print(f"    Final position: {state['pos']}")
    print(f"    Frames: {len(frames)}, Duration: {len(frames)/fps:.1f}s")
    
    return output_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Render connectome-driven flight video')
    parser.add_argument('--duration', type=float, default=1.5, help='Sim duration (s)')
    parser.add_argument('--camera', type=str, default='walker/track2', help='Camera name')
    parser.add_argument('--disturbance', type=str, default='roll_kick', 
                        choices=['none', 'roll_kick', 'pitch_kick', 'yaw_kick'])
    parser.add_argument('--kick', type=float, default=6.0, help='Kick magnitude rad/s')
    parser.add_argument('--nodes', type=str, default='circuits/comprehensive_mobility_nodes.feather', help='Path to nodes feather')
    parser.add_argument('--edges', type=str, default='circuits/comprehensive_mobility_edges.feather', help='Path to edges feather')
    parser.add_argument('--output', type=str, default='visualizations/comprehensive_mobility_flight.mp4')
    args = parser.parse_args()
    
    render_flight_video(
        output_path=args.output,
        nodes_path=args.nodes,
        edges_path=args.edges,
        duration=args.duration,
        camera_name=args.camera,
        disturbance_type=args.disturbance,
        disturbance_magnitude=args.kick,
    )
