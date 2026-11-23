"""
Test 3D acrobatic multi-loop trajectory matching the user's uploaded trajectory.
Features:
1. Ground liftoff jump and smooth ascent to lower deck (Z: 0.05m -> 0.35m)
2. Lower horizontal loop (yawing turn)
3. Ascending helical corkscrew climbing from Z=0.35m to Z=0.75m
4. High-altitude apex sweeping loop with heavy banking ('bending over')
5. Diagonal high-speed descent dive (Z: 0.75m -> 0.28m)
6. Low-altitude high-curvature turnaround loop (Z ~ 0.25m) without crashing
7. Diagonal re-ascent and figure-eight upper loop (Z: 0.25m -> 0.65m)
8. Terminal destination station-keeping hover
"""
import os
import sys
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WORKSPACE_ROOT)

os.environ['MUJOCO_GL'] = 'egl'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from flybody.tasks.trajectory_loaders import constant_speed_trajectory, InferenceFlightTrajectoryLoader
from orthocop.flybody.closed_loop import ClosedLoopFlyBodySim


def generate_acrobatic_multiloop_trajectory(task_dt: float = 0.0002):
    """
    Construct a smooth 3D acrobatic multi-loop flight trajectory matching the
    user's trajectory image (helical climbs, high-altitude apex loop with banking,
    diagonal descent dive, low-altitude figure-8 turnaround loop, and diagonal climb).
    """
    # Define acrobatic segments:
    # (name, duration_s, speed_m_s, yaw_speed_rad_s, bank_deg, d_heading_rad, start_z, end_z)
    segments = [
        # 1. Ground launch & liftoff climb
        ('1. GROUND LAUNCH & ASCENT',       0.45, 0.00,  0.00,  0.0, 0.0, 0.05, 0.35),
        # 2. Lower entry loop
        ('2. LOWER LOOP ENTRY',             0.80, 0.55,  2.20,  8.0, 0.0, 0.35, 0.40),
        # 3. Helical ascending corkscrew 1
        ('3. HELICAL CORKSCREW ASCENT 1',   0.90, 0.65,  2.40, 12.0, 0.0, 0.40, 0.58),
        # 4. Helical ascending corkscrew 2
        ('4. HELICAL CORKSCREW ASCENT 2',   0.90, 0.70,  2.40, 15.0, 0.0, 0.58, 0.76),
        # 5. Apex wide sweeping loop ('bending over')
        ('5. APEX SWEEPING LOOP (HEAVY BANK)', 1.10, 0.60, -2.50,-24.0, 0.0, 0.76, 0.78),
        # 6. Diagonal descent dive (crossing arena)
        ('6. DIAGONAL DESCENT DIVE',        0.90, 0.85,  0.60,  6.0, np.deg2rad(25.0), 0.78, 0.32),
        # 7. Low-altitude turnaround loop (near floor)
        ('7. LOW-ALTITUDE TURNAROUND LOOP', 1.00, 0.50,  2.60, 18.0, 0.0, 0.32, 0.25),
        # 8. Diagonal climb & figure-8 crossover
        ('8. DIAGONAL CLIMB & CROSSOVER',   0.95, 0.75, -1.80,-14.0, np.deg2rad(-20.0), 0.25, 0.62),
        # 9. Upper re-entry loop
        ('9. UPPER RE-ENTRY LOOP',          0.85, 0.55, -2.20,-12.0, 0.0, 0.62, 0.65),
        # 10. Destination station hover
        ('10. DESTINATION STATION HOVER',   0.75, 0.00,  0.00,  0.0, 0.0, 0.65, 0.65)
    ]

    total_duration = sum(s[1] for s in segments)
    total_steps = int(np.ceil(total_duration / task_dt)) + 800

    qpos = np.zeros((total_steps, 7), dtype=np.float64)
    qvel = np.zeros((total_steps, 6), dtype=np.float64)

    cur_step = 0
    cur_pos = np.array([0.0, 0.0, segments[0][6]])
    cur_heading = 0.0

    zone_intervals = []
    t_accum = 0.0

    pitch_y = np.deg2rad(-47.5)

    for seg_idx, (name, dur, sp, yw, bk, hd, z_start, z_end) in enumerate(segments):
        n_seg = int(round(dur / task_dt))
        end_idx = min(cur_step + n_seg, total_steps)
        act_len = end_idx - cur_step
        if act_len <= 0:
            continue

        # 1. Base horizontal path using constant_speed_trajectory
        p_qpos, p_qvel = constant_speed_trajectory(
            n_steps=n_seg,
            speed=sp,
            yaw_speed=yw,
            init_pos=(cur_pos[0], cur_pos[1], z_start),
            init_heading=cur_heading + hd,
            body_rot_angle_y=-47.5,
            body_rot_angle_x=bk,
            control_timestep=task_dt
        )

        # 2. Smooth vertical altitude S-curve
        t_norm = np.linspace(0, 1, n_seg)
        # S-curve: 3*t^2 - 2*t^3
        s_curve = 3 * t_norm**2 - 2 * t_norm**3
        seg_z = z_start + (z_end - z_start) * s_curve
        seg_vz = np.gradient(seg_z, task_dt)

        p_qpos[:, 2] = seg_z
        p_qvel[:, 2] = seg_vz

        qpos[cur_step:end_idx] = p_qpos[:act_len]
        qvel[cur_step:end_idx] = p_qvel[:act_len]

        cur_pos = qpos[end_idx - 1, :3].copy()
        cur_step = end_idx
        cur_heading += (hd + yw * dur)

        zone_intervals.append((name, t_accum, t_accum + dur, (z_start, z_end)))
        t_accum += dur

    # Pad remaining buffer
    while cur_step < total_steps:
        qpos[cur_step] = qpos[cur_step - 1]
        qvel[cur_step] = 0.0
        cur_step += 1

    return qpos, qvel, total_duration, zone_intervals


if __name__ == '__main__':
    qpos, qvel, dur, intervals = generate_acrobatic_multiloop_trajectory()
    print(f'[+] Generated acrobatic trajectory: {dur:.2f}s, {len(qpos)} steps')
    print(f'    X bounds: [{np.min(qpos[:, 0]):.3f}, {np.max(qpos[:, 0]):.3f}] m')
    print(f'    Y bounds: [{np.min(qpos[:, 1]):.3f}, {np.max(qpos[:, 1]):.3f}] m')
    print(f'    Z bounds: [{np.min(qpos[:, 2]):.3f}, {np.max(qpos[:, 2]):.3f}] m')
    for z_name, t0, t1, z_rng in intervals:
        print(f'    • {z_name:<34} ({t0:4.2f}s -> {t1:4.2f}s) | Z: {z_rng[0]:.2f}m -> {z_rng[1]:.2f}m')

