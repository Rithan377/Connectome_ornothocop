"""
FlyBody MuJoCo Flight Environment Wrapper.

Provides high-level physical interface to FlyBody (DeepMind/Janelia fruit fly model),
handling sensor extraction (halteres, gyro, vision, wing posture), actuator commands,
and environmental disturbance injection (wind, gusts, torque kicks, mass variations).
"""

import os
from typing import Dict, Any, Tuple, Optional
import numpy as np

# Silence TF / oneDNN info logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import tf_keras
import tensorflow_probability.python.distributions as tfd
from tensorflow.python.framework import type_spec_registry

# Register legacy TypeSpec alias for FlyBody pretrained policy loader
try:
    _ind = tfd.Independent(tfd.Normal(loc=[0.], scale=[1.]), 1)
    _spec_cls = type(_ind._type_spec)
    type_spec_registry._NAME_TO_TYPE_SPEC[
        'tensorflow_probability.python.distributions.independent.Independent_ACTTypeSpec'
    ] = _spec_cls
except Exception:
    pass

import flybody.fly_envs as fe


class FlyBodyFlightEnv:
    """
    MuJoCo FlyBody Flight Simulation Environment.
    Wraps dm_control/MuJoCo FruitFly with articulated flapping wings and aerodynamics.
    """

    def __init__(
        self,
        ref_path: Optional[str] = 'data/flybody_data/flight-dataset_saccade-evasion_augmented.hdf5',
        wpg_pattern_path: Optional[str] = 'data/flybody_data/wing_pattern_fmech.npy',
        time_limit: float = 2.0,
        random_state: Optional[np.random.RandomState] = None
    ):
        """
        Initialize FlyBody flight simulation.
        
        Args:
            ref_path: Path to flight trajectory reference dataset.
            wpg_pattern_path: Path to experimental wing kinematic pattern.
            time_limit: Maximum flight episode duration (seconds).
            random_state: Seeded RandomState.
        """
        if not os.path.exists(ref_path):
            ref_path = None
        if not os.path.exists(wpg_pattern_path):
            wpg_pattern_path = None

        self.env = fe.flight_imitation(
            ref_path=ref_path,
            wpg_pattern_path=wpg_pattern_path,
            force_actuators=False,
            disable_legs=True,
            random_state=random_state,
            terminal_com_dist=float('inf')
        )
        # Enable continuous flight (1 hour limit avoids float('inf') overflow)
        flight_limit = 3600.0
        self.env._time_limit = flight_limit
        if hasattr(self.env, '_task') and hasattr(self.env._task, '_time_limit'):
            self.env._task._time_limit = flight_limit
        self.time_limit = flight_limit

        # Patch check_termination and before_step so ground takeoff and continuous
        # indefinite flight run smoothly without termination or index out-of-bounds.
        import types
        self._allow_ground = False
        self._custom_trajectory = False
        self.target_z = 0.55
        self.flight_mode = 'evasion'

        orig_check = self.env._task.check_termination
        def safe_check_termination(task_self, physics):
            if self._allow_ground:
                t = physics.time()
                height = physics.data.qpos[2]
                # During initial ground launch phase (t < 0.15s), allow low altitude
                if t < 0.15:
                    return height < 0.015
                # Airborne flight: crash if below floor or inverted
                quat = physics.data.qpos[3:7]
                roll = np.arctan2(2 * (quat[0]*quat[1] + quat[2]*quat[3]), 1 - 2*(quat[1]**2 + quat[2]**2))
                return height < 0.025 or np.abs(roll) > np.deg2rad(120.0)
            return orig_check(physics)
        self.env._task.check_termination = types.MethodType(safe_check_termination, self.env._task)

        orig_before_step = self.env._task.before_step
        def safe_before_step(task_self, physics, action, random_state):
            base_freq, rel_range = task_self._wbpg.base_beat_freq, task_self._wbpg.rel_freq_range
            act = action[task_self._user_idx_action]
            ctrl_freq = base_freq * (1 + rel_range * act)
            ctrl = task_self._wbpg.step(ctrl_freq=ctrl_freq)
            
            length = physics.bind(task_self._wing_joints).qpos
            action[task_self._wing_inds_action] += (ctrl - length)
            
            future_steps = getattr(task_self, '_future_steps', 5)
            max_idx = max(0, len(task_self._ref_qpos) - future_steps - 2)
            step = min(int(np.round(physics.data.time / task_self.control_timestep)), max_idx)
            task_self._step_counter = min(task_self._step_counter, max_idx)
            
            ghost_qpos = task_self._ref_qpos[step, :] + np.hstack((task_self._ghost_offset, 4 * [0]))
            task_self._ghost.set_pose(physics, ghost_qpos[:3], ghost_qpos[3:])
            task_self._ghost.set_velocity(physics, task_self._ref_qvel[step, :3], task_self._ref_qvel[step, 3:])
            
            super(type(task_self), task_self).before_step(physics, action, random_state)
            task_self._step_counter = min(task_self._step_counter, max_idx)
        self.env._task.before_step = types.MethodType(safe_before_step, self.env._task)

        self.current_timestep = None
        self.reset()

    @property
    def physics(self):
        return self.env.physics

    @property
    def dt_ctrl(self):
        return float(self.env.control_timestep())  # 0.0002s (200 µs)

    def configure_flight_mode(
        self,
        mode: str = 'hover',
        speed: Optional[float] = None,
        yaw_speed: Optional[float] = None,
        target_z: float = 0.55,
        heading_deg: float = 0.0,
        bank_deg: Optional[float] = None,
        duration: float = 10.0,
        start_on_ground: bool = True,
        climb_duration: float = 0.40
    ):
        """
        Configure dynamic omnidirectional flight trajectories:
          - 'hover': Stationary mid-air hovering with active altitude hold
          - 'cruise' or 'fast_cruise': High-speed forward flight (0.70 m/s = 70 cm/s)
          - 'loop' or 'turn': Continuous turning circles / loops in mid-air
          - 'diagonal': High-speed diagonal banking flight
        """
        from flybody.tasks.trajectory_loaders import constant_speed_trajectory, InferenceFlightTrajectoryLoader
        
        task_dt = float(self.env.control_timestep())
        # Cap pre-allocated trajectory to at most 25.0s (125,000 steps).
        # safe_before_step safely clamps reference indexing for t > 25s, enabling
        # instantaneous sub-second launch and infinite continuous flight without latency.
        gen_duration = min(duration, 25.0) if duration > 0 else 25.0
        total_steps = int(np.ceil(gen_duration / task_dt)) + 600  # extra padding avoids slicing truncations
        self.target_z = target_z
        self.flight_mode = mode
        self._allow_ground = start_on_ground
        self._custom_trajectory = True

        # Mode parameter presets
        if mode == 'hover':
            sp = 0.0 if speed is None else speed
            yw = 0.0 if yaw_speed is None else yaw_speed
            bk = 0.0 if bank_deg is None else bank_deg
            hd = np.deg2rad(heading_deg)
        elif mode in ('cruise', 'fast_cruise'):
            sp = 0.70 if speed is None else speed      # 70 cm/s = 0.7 m/s fast biological cruise
            yw = 0.0 if yaw_speed is None else yaw_speed
            bk = 0.0 if bank_deg is None else bank_deg
            hd = np.deg2rad(heading_deg)
        elif mode in ('loop', 'turn'):
            sp = 0.40 if speed is None else speed      # 40 cm/s circling speed
            yw = 2.2 if yaw_speed is None else yaw_speed  # 2.2 rad/s turning rate
            bk = 0.0 if bank_deg is None else bank_deg
            hd = np.deg2rad(heading_deg)
        elif mode == 'diagonal':
            sp = 0.60 if speed is None else speed      # 60 cm/s diagonal speed
            yw = 0.0 if yaw_speed is None else yaw_speed
            bk = 10.0 if bank_deg is None else bank_deg   # 10° right bank
            hd = np.deg2rad(35.0 if heading_deg == 0.0 else heading_deg)
        elif mode == 'exploratory':
            from scripts.render_exploratory_flight import generate_exploratory_mission_trajectory
            qpos, qvel, total_duration, _ = generate_exploratory_mission_trajectory(task_dt)
            loader = InferenceFlightTrajectoryLoader()
            loader.set_next_trajectory(qpos, qvel)
            self.env._task._traj_generator = loader
            self.env._task._time_limit = total_duration + 3600.0
            self.env._time_limit = total_duration + 3600.0
            return
        else:
            sp = 0.0 if speed is None else speed
            yw = 0.0 if yaw_speed is None else yaw_speed
            bk = 0.0 if bank_deg is None else bank_deg
            hd = np.deg2rad(heading_deg)

        qpos, qvel = constant_speed_trajectory(
            n_steps=total_steps,
            speed=sp,
            yaw_speed=yw,
            init_pos=(0, 0, target_z),
            init_heading=hd,
            body_rot_angle_y=-47.5,
            body_rot_angle_x=bk,
            control_timestep=task_dt
        )

        if start_on_ground:
            ground_z = 0.05
            n_climb = int(climb_duration / task_dt)
            t_norm = np.linspace(0, 1, n_climb)
            s_curve = 3 * t_norm**2 - 2 * t_norm**3
            climb_z = ground_z + (target_z - ground_z) * s_curve
            
            qpos[:n_climb, 2] = climb_z
            qpos[n_climb:, 2] = target_z
            qvel[:n_climb, 2] = np.gradient(climb_z, task_dt)
            qvel[n_climb:, 2] = 0.0
            
            ramp_sp = np.linspace(0.0, sp, n_climb)
            qvel[:n_climb, 0] = ramp_sp * np.cos(hd)
            qvel[:n_climb, 1] = ramp_sp * np.sin(hd)

        loader = InferenceFlightTrajectoryLoader()
        loader.set_next_trajectory(qpos, qvel)
        self.env._task._traj_generator = loader
        self.env._task._time_limit = duration + 3600.0
        self.env._time_limit = duration + 3600.0

    def reset(self, start_on_ground: bool = False) -> Dict[str, Any]:
        """Reset environment to initial flight trim pose or ground stance."""
        self._allow_ground = start_on_ground
        self.current_timestep = self.env.reset()
        if start_on_ground:
            self.set_ground_stance(ground_z=0.05, pitch_deg=-47.5)
        return self.get_state()

    def set_ground_stance(self, ground_z: float = 0.05, pitch_deg: float = -47.5):
        """Place fly in grounded stance on the floor with natural nose-up takeoff pitch."""
        # Enable collision with groundplane
        self.physics.model.geom_contype[0] = 1
        self.physics.model.geom_conaffinity[0] = 1
        for i in range(self.physics.model.ngeom):
            name = self.physics.model.id2name(i, 'geom')
            if name and 'collision' in name.lower():
                self.physics.model.geom_contype[i] = 1
                self.physics.model.geom_conaffinity[i] = 1

        theta = np.deg2rad(pitch_deg)
        q_pitch = np.array([np.cos(theta/2), 0.0, np.sin(theta/2), 0.0])
        
        self.physics.data.qpos[0:3] = [0.0, 0.0, ground_z]
        self.physics.data.qpos[3:7] = q_pitch
        self.physics.data.qvel[:] = 0.0
        self.physics.forward()

        # Adapt reference trajectory only if not using custom trajectory generator
        if not self._custom_trajectory and hasattr(self.env._task, '_ref_qpos'):
            target_z = 0.55  # 55 cm cruise altitude
            n_ramp = min(1500, len(self.env._task._ref_qpos))
            ramp = np.linspace(ground_z, target_z, n_ramp)
            self.env._task._ref_qpos[:n_ramp, 2] = ramp
            self.env._task._ref_qpos[n_ramp:, 2] = target_z

    def trigger_takeoff_jump(self, jump_speed: float = 1.2, forward_speed: float = 0.0):
        """
        Simulate Giant Fiber / TTMn explosive middle leg extension jump reflex.
        Catapults the fly off the ground into the air.
        jump_speed: 1.2 m/s (biological Drosophila jump takeoff velocity).
        """
        self.physics.data.qvel[2] = jump_speed
        self.physics.data.qvel[0] = forward_speed
        self.physics.forward()
        self.physics.forward()

    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool]:
        """
        Step physical MuJoCo simulation with 12-dimensional action vector.
        
        Action spec (12 channels):
          [0:3]   head: abduct, twist, pitch
          [3:6]   left wing: yaw, roll, pitch
          [6:9]   right wing: yaw, roll, pitch
          [9:11]  abdomen: abduct, pitch
          [11]    user_0: wingbeat frequency modulation via WBPG
          
        Returns:
            (state_dict, reward, is_terminal)
        """
        action = np.clip(action, self.env.action_spec().minimum, self.env.action_spec().maximum)
        self.current_timestep = self.env.step(action)
        reward = float(self.current_timestep.reward) if self.current_timestep.reward is not None else 0.0
        done = self.current_timestep.last()
        state = self.get_state()
        return state, reward, done

    def get_state(self) -> Dict[str, Any]:
        """
        Extract complete physical measurements from MuJoCo physics.
        """
        obs = self.current_timestep.observation
        
        # Root freejoint state (position, orientation quaternion, velocities)
        qpos = self.physics.data.qpos[:7].copy()
        qvel = self.physics.data.qvel[:6].copy()
        
        pos = qpos[:3]              # [x, y, z] in world frame (meters)
        quat = qpos[3:7]            # [qw, qx, qy, qz] orientation quaternion
        lin_vel = qvel[:3]          # [vx, vy, vz] world frame (m/s)
        ang_vel = obs['walker/gyro'].copy()  # [p, q, r] body frame gyro (rad/s)
        lin_acc = obs['walker/accelerometer'].copy() # [ax, ay, az] (m/s^2)
        world_z = obs['walker/world_zaxis'].copy()   # gravity vector projection
        
        # Calculate Euler angles (roll phi, pitch theta, yaw psi) from quaternion
        qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
        # Roll (x-axis)
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        # Pitch (y-axis)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        # Yaw (z-axis)
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        
        sim_time = float(self.physics.data.time)

        return {
            'time': sim_time,
            'pos': pos,
            'quat': quat,
            'euler': np.array([roll, pitch, yaw]), # [phi, theta, psi] in radians
            'lin_vel': lin_vel,
            'ang_vel': ang_vel,                     # [p, q, r] in rad/s
            'lin_acc': lin_acc,
            'world_z': world_z,
            'joints_pos': obs['walker/joints_pos'].copy(),
            'joints_vel': obs['walker/joints_vel'].copy(),
            'raw_observation': obs
        }

    def apply_torque_kick(self, roll_kick: float = 0.0, pitch_kick: float = 0.0, yaw_kick: float = 0.0):
        """
        Apply instantaneous impulsive angular disturbance to body (simulating sudden gust or collision).
        """
        # MuJoCo qvel[3:6] corresponds to root body rotational velocity [p, q, r]
        self.physics.data.qvel[3] += roll_kick
        self.physics.data.qvel[4] += pitch_kick
        self.physics.data.qvel[5] += yaw_kick

    def apply_wind_force(self, wind_force_vector: np.ndarray):
        """
        Apply external aerodynamic wind force to thorax body.
        """
        # Apply external force (3D) to root thorax body (body index 2)
        thorax_id = self.physics.model.name2id('walker/thorax', 'body')
        self.physics.data.xfrc_applied[thorax_id, :3] = wind_force_vector

    def perturb_mass(self, mass_factor: float = 1.0):
        """
        Scale total fly body mass to test inertia robustness.
        """
        self.physics.model.body_mass[1:] *= mass_factor
