#!/usr/bin/env python3
"""
Interactive Native MuJoCo Flight Simulator with Drosophila Connectome.

Launches the official native MuJoCo GLFW viewer window directly on the desktop
running the exact Connectome-Hybrid flight controller from `connectome_mujoco_flight.mp4`.
"""

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
import time
import argparse
from typing import Optional, Dict, Any, List
import numpy as np
import tensorflow as tf

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

import mujoco
import mujoco.viewer

from orthocop.flybody.closed_loop import ClosedLoopFlyBodySim


def sync_to_viewer(src_d, dst_d, model):
    """Synchronize physical state from dm_control physics to viewer MjData."""
    np.copyto(dst_d.qpos, src_d.qpos)
    np.copyto(dst_d.qvel, src_d.qvel)
    if model.na > 0 and len(dst_d.act) == len(src_d.act):
        np.copyto(dst_d.act, src_d.act)
    if model.nmocap > 0 and len(dst_d.mocap_pos) == len(src_d.mocap_pos):
        np.copyto(dst_d.mocap_pos, src_d.mocap_pos)
        np.copyto(dst_d.mocap_quat, src_d.mocap_quat)
    dst_d.time = src_d.time
    mujoco.mj_forward(model, dst_d)


def render_trajectory(viewer, trajectory, stride=1):
    """
    Render complete 3D flight trajectory in MuJoCo viewer using cyan capsules.
    Directly implements user's MuJoCo trajectory visualizer.
    """
    cyan = np.array([0.10, 0.95, 0.85, 1.0], dtype=np.float32)
    eye9 = np.eye(3).flatten()
    zeros3 = np.zeros(3)
    scene = viewer.user_scn
    with viewer.lock():
        scene.ngeom = 0
        points = trajectory[::stride]
        for i in range(len(points) - 1):
            if scene.ngeom >= scene.maxgeom:
                break
            p1 = points[i]
            p2 = points[i + 1]
            geom = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                zeros3,
                zeros3,
                eye9,
                cyan
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                0.0020,  # 2.0mm ribbon capsule for spaced-out path
                p1,
                p2
            )
            geom.rgba[:] = cyan
            scene.ngeom += 1


def run_interactive_mujoco(
    duration: float = 0.0,
    connectome_blend: float = 0.35,
    auto_perturb: bool = False,
    perturb_interval: float = 5.0,
    perturb_strength: float = 5.0,
    brain_type: str = 'comprehensive',
    takeoff: bool = True,
    mode: str = 'hover',
    speed: Optional[float] = None,
):
    print("=" * 75)
    print("   DROSOPHILA CONNECTOME — NATIVE MUJOCO 3D FLIGHT SIMULATOR")
    if brain_type == 'comprehensive':
        nodes_path = 'circuits/comprehensive_mobility_nodes.feather'
        edges_path = 'circuits/comprehensive_mobility_edges.feather'
        print("   [Comprehensive Mobility Brain: 10,668 neurons | 527K synapses]")
    else:
        nodes_path = 'circuits/baseline_flight_brain_nodes.feather'
        edges_path = 'circuits/baseline_flight_brain_edges.feather'
        print("   [Baseline Flight Brain: 897 neurons | 34K synapses]")
    print(f"   [Flight Mode: {mode.upper()} | Start Ground Takeoff: {takeoff} | Blend: {connectome_blend:.2f}]")
    print("=" * 75)
    print(f"[*] [1/4] Loading Connectome brain & neural circuits ({brain_type.capitalize()})...", flush=True)
    t_start = time.time()
    
    sim = ClosedLoopFlyBodySim(
        controller_type='connectome_hybrid',
        nodes_path=nodes_path,
        edges_path=edges_path,
        connectome_blend=connectome_blend
    )
    print(f"[*] [2/4] Initializing FlyBody physical dynamics & sensors ({time.time() - t_start:.2f}s)...", flush=True)
    env = sim.env
    m = env.physics.model.ptr
    d_viewer = mujoco.MjData(m)

    target_z = 0.55
    ribbon_points = None
    ref_banks = None
    ref_psis = None

    if mode == 'acrobatic':
        from scripts.run_acrobatic_flight_simulation import generate_scaled_flight_trajectory
        from flybody.tasks.trajectory_loaders import InferenceFlightTrajectoryLoader
        print("[*] Generating spaced-out 42-point 3D B-spline trajectory for MuJoCo visualizer...", flush=True)
        qpos_acro, qvel_acro, total_dur, _, ref_banks, ref_psis, ribbon_points, start_xyz = generate_scaled_flight_trajectory(
            task_dt=env.dt_ctrl, v_cruise=0.85
        )
        loader = InferenceFlightTrajectoryLoader()
        loader.set_next_trajectory(qpos_acro, qvel_acro)
        env.env._task._traj_generator = loader
        env.env._task._time_limit = total_dur + 3600.0
        env.env._time_limit = total_dur + 3600.0
        sim.decoder.use_ref_tracking = False
    else:
        print(f"[*] [3/4] Configuring '{mode.upper()}' flight trajectory & motor decoder...", flush=True)
        env.configure_flight_mode(
            mode=mode,
            speed=speed,
            target_z=target_z,
            duration=25.0,
            start_on_ground=takeoff,
            climb_duration=0.40
        )

    if mode in ('exploratory', 'acrobatic'):
        sim.decoder.use_ref_tracking = False
        import types
        task = env.env._task
        def safe_term(task_self, physics):
            height = physics.data.qpos[2]
            t = physics.time()
            if t > 0.80 and height < 0.025:
                return True
            return False
        task.check_termination = types.MethodType(safe_term, task)

    class AcrobaticBalancedDecoder:
        """
        Ultra-Smooth Biological Motor Decoder v2 with:
        - 2nd-order Butterworth LPF (12Hz cutoff) on angular velocity
        - Extended haltere gyro filter (τ=20ms), vertical velocity filter (τ=30ms)
        - Extended neuromuscular smoothing (τ=18ms)
        - Reduced PD gains to eliminate controller-induced wobble
        - Integral altitude controller with anti-windup
        """
        def __init__(self, sim, task_dt: float = 0.0002):
            self.sim = sim
            self.dt = task_dt
            self.alpha_gyro = task_dt / (0.020 + task_dt)
            self.alpha_vz = task_dt / (0.030 + task_dt)
            self.alpha_act = task_dt / (0.018 + task_dt)

            # 2nd-order Butterworth LPF coefficients (12Hz cutoff at 5000Hz)
            import math
            fc = 12.0
            fs = 1.0 / task_dt
            omega_c = 2.0 * math.pi * fc / fs
            omega_p = math.tan(omega_c / 2.0)
            omega_p2 = omega_p * omega_p
            sqrt2 = math.sqrt(2.0)
            denom = 1.0 + sqrt2 * omega_p + omega_p2
            self.bw_b0 = omega_p2 / denom
            self.bw_b1 = 2.0 * self.bw_b0
            self.bw_b2 = self.bw_b0
            self.bw_a1 = 2.0 * (omega_p2 - 1.0) / denom
            self.bw_a2 = (1.0 - sqrt2 * omega_p + omega_p2) / denom
            self.bw_z1 = np.zeros(3)
            self.bw_z2 = np.zeros(3)

            self.alt_integral = 0.0
            self.alt_int_max = 0.35
            self.filt_ang_vel = np.zeros(3)
            self.filt_vz = 0.0
            self.prev_action = np.zeros(12)

        def reset(self):
            self.filt_ang_vel = np.zeros(3)
            self.bw_z1 = np.zeros(3)
            self.bw_z2 = np.zeros(3)
            self.filt_vz = 0.0
            self.prev_action = np.zeros(12)
            self.alt_integral = 0.0

        def _butterworth2_step(self, x):
            y = self.bw_b0 * x + self.bw_z1
            self.bw_z1 = self.bw_b1 * x - self.bw_a1 * y + self.bw_z2
            self.bw_z2 = self.bw_b2 * x - self.bw_a2 * y
            return y

        def decode(self, motor_rates, current_euler, state=None, target_z=None, bank_ref=0.0, psi_ref=0.0, t=0.0):
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
            
            raw_ang_vel = state.get('ang_vel', np.zeros(3)) if state else np.zeros(3)
            self.filt_ang_vel = (1.0 - self.alpha_gyro) * self.filt_ang_vel + self.alpha_gyro * raw_ang_vel
            smooth_ang_vel = self._butterworth2_step(self.filt_ang_vel)

            raw_vz = state.get('lin_vel', np.zeros(3))[2] if state else 0.0
            self.filt_vz = (1.0 - self.alpha_vz) * self.filt_vz + self.alpha_vz * raw_vz
            
            roll_err = current_euler[0] - bank_ref
            pitch_err = current_euler[1] - np.deg2rad(-47.5)
            yaw_err = current_euler[2] - psi_ref
            
            pd_roll  = -0.16 * roll_err  - 0.008 * smooth_ang_vel[0]
            pd_pitch = -0.16 * pitch_err - 0.007 * smooth_ang_vel[1]
            pd_yaw   = -0.006 * smooth_ang_vel[2]
            
            action = np.zeros(12)
            action[0] = np.clip(-0.12 * yaw_err, -0.10, 0.10)
            action[1] = np.clip(-0.18 * roll_err, -0.25, 0.25)
            action[2] = np.clip(-0.12 * pitch_err, -0.08, 0.08)

            action[3] = np.clip(conn_yaw + pd_roll, -1.0, 1.0)
            action[4] = np.clip(conn_roll + pd_pitch, -1.0, 1.0)
            action[5] = np.clip(conn_pitch, -1.0, 1.0)
            action[6] = np.clip(-conn_yaw - pd_roll, -1.0, 1.0)
            action[7] = np.clip(conn_roll + pd_pitch, -1.0, 1.0)
            action[8] = np.clip(-conn_pitch, -1.0, 1.0)
            action[9] = np.clip(0.35 * pd_yaw, -0.5, 0.5)
            
            delta_freq = self.sim.decoder.k_freq * (dlm_mean - dvm_mean)
            if target_z is not None and state is not None:
                alt_err = target_z - state['pos'][2]
                cos_roll = max(0.5, np.cos(current_euler[0]))
                bank_comp = (1.0 / cos_roll - 1.0) * 0.25
                self.alt_integral += alt_err * self.dt
                self.alt_integral = np.clip(self.alt_integral, -self.alt_int_max, self.alt_int_max)
                delta_freq += (3.5 * alt_err
                             + 1.8 * self.alt_integral
                             - 0.015 * self.filt_vz
                             + 0.16
                             + bank_comp)
                if state['pos'][2] < 0.12 and t > 0.35:
                    delta_freq += 0.45
            action[11] = np.clip(delta_freq, -1.0, 1.0)
            
            smoothed = (1.0 - self.alpha_act) * self.prev_action + self.alpha_act * action
            self.prev_action = smoothed.copy()
            return smoothed

    acro_decoder = AcrobaticBalancedDecoder(sim, env.dt_ctrl)

    state = env.reset(start_on_ground=takeoff)
    if takeoff:
        env.set_ground_stance(ground_z=0.05, pitch_deg=-47.5)
        env.trigger_takeoff_jump(jump_speed=1.2, forward_speed=0.0)
    if sim.circuit is not None:
        sim.circuit.reset()
    sync_to_viewer(env.physics.data.ptr, d_viewer, m)

    # Find thorax body id for camera tracking
    thorax_id = 2
    for i in range(m.nbody):
        name = env.physics.model.id2name(i, 'body')
        if name and ('thorax' in name or 'walker/thorax' in name):
            thorax_id = i
            break

    print(f"[*] [4/4] Opening native MuJoCo interactive GUI window (Ready in {time.time() - t_start:.2f}s total)...", flush=True)
    print(f"    • Camera tracking target: body '{env.physics.model.id2name(thorax_id, 'body')}' (id={thorax_id})")
    print("[*] Launching native MuJoCo interactive GUI window...")
    print("    • Viewport: Auto-tracking camera centered on fly")
    print("    • Spacebar / Run / Pause: Toggles simulation play/pause")
    print("    • Left Mouse Drag: Rotate 3D View")
    print("    • Right Mouse Drag: Pan Camera / Move")
    print("    • Scroll Wheel: Zoom In / Out")
    print("    • Ctrl + Right-Click Drag: Grab & tug any body part with physical force!")
    print("    • Ctrl + Left-Click Drag: Apply torque / twist fly")
    print("    • Backspace: Reset Fly")
    print("    • Close window or Ctrl+C in terminal to exit")
    print("-" * 70)

    with mujoco.viewer.launch_passive(m, d_viewer) as viewer:
        sim_ctrl = viewer._get_sim()

        # Render 3D flight trajectory ribbon if available
        if ribbon_points is not None:
            render_trajectory(viewer, ribbon_points, stride=1)

        # Camera configuration: close-up tracking view of the fruit fly in mid-air
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        viewer.cam.trackbodyid = thorax_id
        viewer.cam.distance = 0.10 if mode == 'acrobatic' else 0.05
        viewer.cam.elevation = -12.0
        viewer.cam.azimuth = 135.0
        viewer.cam.lookat[0] = d_viewer.xpos[thorax_id, 0]
        viewer.cam.lookat[1] = d_viewer.xpos[thorax_id, 1]
        viewer.cam.lookat[2] = d_viewer.xpos[thorax_id, 2]

        step_idx = 0
        episode_step = 0
        dt = env.dt_ctrl
        last_perturb_time = 0.0
        sim_time = 0.0

        perturb_types = ['roll_kick', 'pitch_kick', 'yaw_kick']
        perturb_idx = 0

        while viewer.is_running():
            step_start = time.time()

            # Respect GUI Pause / Run state
            if not sim_ctrl.run:
                viewer.sync()
                time.sleep(0.02)
                continue

            sim_time = step_idx * dt

            # Optional perturbation testing
            if auto_perturb and (sim_time - last_perturb_time >= perturb_interval) and sim_time >= 2.0:
                ptype = perturb_types[perturb_idx % len(perturb_types)]
                if ptype == 'roll_kick':
                    env.apply_torque_kick(roll_kick=perturb_strength)
                    print(f"\n[⚡ PERTURBATION] Injected Roll Kick ({perturb_strength:.1f} rad/s) at t={sim_time:.2f}s")
                elif ptype == 'pitch_kick':
                    env.apply_torque_kick(pitch_kick=perturb_strength)
                    print(f"\n[⚡ PERTURBATION] Injected Pitch Kick ({perturb_strength:.1f} rad/s) at t={sim_time:.2f}s")
                elif ptype == 'yaw_kick':
                    env.apply_torque_kick(yaw_kick=perturb_strength)
                    print(f"\n[⚡ PERTURBATION] Injected Yaw Kick ({perturb_strength:.1f} rad/s) at t={sim_time:.2f}s")
                last_perturb_time = sim_time
                perturb_idx += 1

            # --- Exact Hybrid Controller Computation ---
            obs_tf = {
                k: tf.convert_to_tensor(np.expand_dims(v, 0), dtype=tf.float32)
                for k, v in state['raw_observation'].items()
            }
            base_action = sim.pretrained_policy(obs_tf).mean()[0].numpy()

            sensory = sim.encoder.update(state)
            motor_rates = sim.circuit.step(sensory)
            if mode == 'acrobatic':
                try:
                    traj_step = min(episode_step, env.env._task._traj_generator._qpos.shape[0] - 1)
                    ref_z = env.env._task._traj_generator._qpos[traj_step, 2]
                    b_ref = ref_banks[traj_step] if ref_banks is not None else 0.0
                    p_ref = ref_psis[traj_step] if ref_psis is not None else 0.0
                except Exception:
                    ref_z = 0.50
                    b_ref = 0.0
                    p_ref = 0.0
                conn_action = acro_decoder.decode(motor_rates, state['euler'], state=state, target_z=ref_z, bank_ref=b_ref, psi_ref=p_ref, t=sim_time)
            else:
                conn_action = sim.decoder.decode(motor_rates, state['euler'], state=state, target_z=target_z)

            action = base_action + connectome_blend * conn_action
            # Altitude channel: higher blend (0.40) for acrobatic Z tracking
            if mode == 'acrobatic':
                action[11] = base_action[11] + 0.40 * conn_action[11]
            action = np.clip(action, env.env.action_spec().minimum, env.env.action_spec().maximum)

            state, reward, done = env.step(action)
            episode_step += 1

            # Sync to viewer
            with viewer.lock():
                sync_to_viewer(env.physics.data.ptr, d_viewer, m)
                viewer.cam.lookat[0] = d_viewer.xpos[thorax_id, 0]
                viewer.cam.lookat[1] = d_viewer.xpos[thorax_id, 1]
                viewer.cam.lookat[2] = d_viewer.xpos[thorax_id, 2]

            viewer.sync()

            # Episode looping: only reset on genuine crash (below floor after launch or inverted)
            if mode in ('exploratory', 'acrobatic'):
                is_crashed = (sim_time > 0.40) and (state['pos'][2] < 0.025)
            else:
                is_crashed = (sim_time > 0.40) and (state['pos'][2] < 0.025 or abs(state['euler'][0]) > np.deg2rad(120.0))
            if done or is_crashed:
                if mode == 'acrobatic':
                    loader.set_next_trajectory(qpos_acro, qvel_acro)
                    state = env.reset(start_on_ground=takeoff)
                else:
                    env.configure_flight_mode(
                        mode=mode,
                        speed=speed,
                        target_z=target_z,
                        duration=25.0,
                        start_on_ground=takeoff,
                        climb_duration=0.40
                    )
                    state = env.reset(start_on_ground=takeoff)
                if sim.circuit is not None:
                    sim.circuit.reset()
                acro_decoder.reset()
                if takeoff:
                    env.set_ground_stance(ground_z=0.05, pitch_deg=-47.5)
                    env.trigger_takeoff_jump(jump_speed=1.2, forward_speed=0.0)
                if ribbon_points is not None:
                    render_trajectory(viewer, ribbon_points, stride=1)
                episode_step = 0
                last_perturb_time = sim_time
                with viewer.lock():
                    sync_to_viewer(env.physics.data.ptr, d_viewer, m)
                viewer.sync()

            # Telemetry display every 100 steps (~20ms)
            if step_idx % 100 == 0:
                euler_deg = state['euler'] * 180.0 / np.pi
                pos = state['pos']
                vel = state['lin_vel']
                speed_cms = np.linalg.norm(vel) * 100.0
                dlm_rate = motor_rates.get('DLM_L', 0)
                steer_rate = np.mean([motor_rates.get('roll_L', 0), motor_rates.get('pitch_L', 0)])
                sys.stdout.write(
                    f"\r[{mode.upper():7s}] t={sim_time:6.2f}s | "
                    f"Alt={pos[2]*100:5.1f}cm | "
                    f"Speed={speed_cms:5.1f}cm/s | "
                    f"R={euler_deg[0]:+5.1f}° P={euler_deg[1]:+5.1f}° Y={euler_deg[2]:+5.1f}° | "
                    f"DLM: {dlm_rate:.0f}Hz | Steer: {steer_rate:.0f}Hz"
                )
                sys.stdout.flush()

            step_idx += 1

            # Real-time synchronization
            elapsed = time.time() - step_start
            sleep_time = dt - elapsed
            if sleep_time > 0.0001:
                time.sleep(sleep_time)

            if duration > 0 and sim_time >= duration:
                print(f"\n[+] Reached duration limit ({duration}s). Continuing indefinitely...")
                duration = 0

    print("\n[*] Native MuJoCo viewer closed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Native MuJoCo Fruit Fly Connectome-Hybrid Simulator')
    parser.add_argument('--duration', type=float, default=0, help='Duration in seconds (0 = infinite)')
    parser.add_argument('--brain', type=str, default='comprehensive', choices=['comprehensive', 'baseline'],
                        help='Brain model: comprehensive (10,668n) or baseline (897n)')
    parser.add_argument('--blend', type=float, default=0.35, help='Connectome blend factor (default 0.35)')
    parser.add_argument('--mode', type=str, default='hover', choices=['hover', 'cruise', 'loop', 'diagonal', 'exploratory', 'acrobatic'],
                        help='Flight maneuver: hover, cruise, loop, diagonal, exploratory, or acrobatic')
    parser.add_argument('--speed', type=float, default=None, help='Flight speed in m/s (default: 0.70 m/s for cruise)')
    parser.add_argument('--auto-perturb', action='store_true', help='Enable automatic periodic perturbation kicks')
    parser.add_argument('--kick', type=float, default=5.0, help='Perturbation kick magnitude (rad/s)')
    parser.add_argument('--no-takeoff', dest='takeoff', action='store_false', help='Disable ground takeoff and start in mid-air')
    parser.set_defaults(takeoff=True)
    args = parser.parse_args()

    run_interactive_mujoco(
        duration=args.duration,
        connectome_blend=args.blend,
        auto_perturb=args.auto_perturb,
        perturb_strength=args.kick,
        brain_type=args.brain,
        takeoff=args.takeoff,
        mode=args.mode,
        speed=args.speed
    )
