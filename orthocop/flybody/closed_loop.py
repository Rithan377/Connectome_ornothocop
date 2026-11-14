"""
Closed-Loop FlyBody MuJoCo + Connectome Spiking Neural Simulation Engine.

Orchestrates the real-time feedback loop between MuJoCo physics and the Male
Drosophila CNS Connectome spiking neural circuit, supporting disturbance injection
and automated telemetry recording.
"""

from typing import Dict, Any, Optional, List
import numpy as np

from orthocop.flybody.flybody_env import FlyBodyFlightEnv
from orthocop.flybody.sensor_encoder import BiophysicalSensorEncoder
from orthocop.flybody.motor_decoder import MotorDecoder
from orthocop.flybody.connectome_brian2 import ConnectomeFlightCircuit


class ClosedLoopFlyBodySim:
    """
    Closed-loop simulation engine integrating FlyBody MuJoCo and Connectome LIF.
    """

    def __init__(
        self,
        controller_type: str = 'connectome_reduced', # 'pretrained', 'connectome_full', 'connectome_reduced', 'connectome_hybrid'
        nodes_path: str = 'circuits/stabilized_flight_nodes.feather',
        edges_path: str = 'circuits/stabilized_flight_edges.feather',
        ablated_nodes: Optional[List[int]] = None,
        ablate_halteres: bool = False,
        ablate_visual: bool = False,
        haltere_delay_ms: float = 1.5,
        haltere_noise_std: float = 0.05,
        connectome_blend: float = 0.35  # Blend factor for hybrid mode (0=pretrained, 1=full connectome)
    ):
        self.controller_type = controller_type
        self.connectome_blend = connectome_blend
        
        # 1. Physical environment
        self.env = FlyBodyFlightEnv()

        # 2. Pretrained policy baseline loader
        self.pretrained_policy = None
        if controller_type in ('pretrained', 'connectome_hybrid'):
            import tensorflow as tf
            self.pretrained_policy = tf.saved_model.load('data/flybody_data/flight')

        # 3. Connectome neural circuit & sensor/motor interfaces
        self.circuit = None
        self.encoder = None
        self.decoder = None

        if controller_type not in ('pretrained',):
            self.circuit = ConnectomeFlightCircuit(
                nodes_path=nodes_path,
                edges_path=edges_path,
                ablated_nodes=set(ablated_nodes) if ablated_nodes else set(),
                ablate_halteres=ablate_halteres,
                ablate_visual=ablate_visual
            )
            self.encoder = BiophysicalSensorEncoder(
                haltere_delay_ms=haltere_delay_ms,
                haltere_noise_std=haltere_noise_std
            )
            self.decoder = MotorDecoder()

    def run_simulation(
        self,
        duration: float = 1.0,
        flight_mode: str = 'hover',
        speed: Optional[float] = None,
        yaw_speed: Optional[float] = None,
        target_z: float = 0.55,
        disturbance_type: str = 'none',
        disturbance_time: float = 0.5,
        disturbance_magnitude: float = 5.0,
        wind_vector: Optional[np.ndarray] = None,
        start_on_ground: bool = True,
        climb_duration: float = 0.40
    ) -> Dict[str, Any]:
        """
        Run closed-loop simulation trajectory.
        
        Args:
            duration: Total duration in seconds.
            flight_mode: 'hover', 'cruise', 'loop', 'diagonal'.
            speed: Optional override forward speed (m/s).
            yaw_speed: Optional override yaw turning speed (rad/s).
            target_z: Target flight altitude (meters).
            disturbance_type: 'none', 'roll_kick', 'pitch_kick', 'yaw_kick',
                              'crosswind', 'gust', 'turbulence'.
            disturbance_time: Time at which disturbance is applied (seconds).
            disturbance_magnitude: Angular velocity kick (rad/s) or wind force.
            wind_vector: Optional constant or time-varying wind force (N).
            start_on_ground: Whether to start grounded on floor and execute takeoff.
            climb_duration: Duration of takeoff climb ascent (seconds).
            
        Returns:
            Dictionary with telemetry history and quantitative performance metrics.
        """
        # Configure flight trajectory
        self.env.configure_flight_mode(
            mode=flight_mode,
            speed=speed,
            yaw_speed=yaw_speed,
            target_z=target_z,
            duration=duration,
            start_on_ground=start_on_ground,
            climb_duration=climb_duration
        )

        state = self.env.reset(start_on_ground=start_on_ground)
        if self.circuit is not None:
            self.circuit.reset()

        # If starting on ground, trigger Giant Fiber / TTM jump takeoff reflex
        if start_on_ground:
            self.env.trigger_takeoff_jump(jump_speed=1.2, forward_speed=0.0)

        dt = self.env.dt_ctrl
        n_steps = int(round(duration / dt))

        # Telemetry logs
        times = []
        positions = []
        euler_angles = []
        ang_velocities = []
        actions = []
        rewards = []
        motor_rates_history = []

        disturbance_applied = False
        peak_disturbance_rate = 0.0
        recovery_time = None
        crashed = False

        # Turbulence state (Ornstein-Uhlenbeck stochastic process)
        turb_state = np.zeros(3)

        for step_idx in range(n_steps):
            t = step_idx * dt
            times.append(t)
            
            pos = state['pos'].copy()
            euler = state['euler'].copy()
            omega = state['ang_vel'].copy()
            
            positions.append(pos)
            euler_angles.append(euler)
            ang_velocities.append(omega)

            # Check crash criteria (altitude drop or complete inversion)
            min_z = 0.025 if start_on_ground else 0.15
            if (not start_on_ground or t > climb_duration) and (pos[2] < min_z or np.abs(euler[0]) > np.deg2rad(120.0)):
                crashed = True

            # 1. Disturbance injection
            if not disturbance_applied and t >= disturbance_time:
                if disturbance_type == 'roll_kick':
                    self.env.apply_torque_kick(roll_kick=disturbance_magnitude)
                    peak_disturbance_rate = disturbance_magnitude
                elif disturbance_type == 'pitch_kick':
                    self.env.apply_torque_kick(pitch_kick=disturbance_magnitude)
                    peak_disturbance_rate = disturbance_magnitude
                elif disturbance_type == 'yaw_kick':
                    self.env.apply_torque_kick(yaw_kick=disturbance_magnitude)
                    peak_disturbance_rate = disturbance_magnitude
                elif disturbance_type == 'gust':
                    self.env.apply_wind_force(np.array([disturbance_magnitude * 1e-4, 0.0, 0.0]))
                disturbance_applied = True

            # Continuous wind / turbulence
            if disturbance_type == 'crosswind':
                self.env.apply_wind_force(np.array([0.0, disturbance_magnitude * 1e-4, 0.0]))
            elif disturbance_type == 'turbulence':
                theta_ou = 2.0
                sigma_ou = 0.05
                turb_state += -theta_ou * turb_state * dt + sigma_ou * np.random.normal(0, np.sqrt(dt), size=3)
                self.env.apply_wind_force(turb_state * 1e-3)

            # Track recovery time after disturbance (rate drops below 10% of peak)
            if disturbance_applied and recovery_time is None and t > disturbance_time:
                curr_rate = np.linalg.norm(omega)
                if curr_rate < 0.10 * peak_disturbance_rate:
                    recovery_time = t - disturbance_time

            # 2. Control computation
            if self.controller_type == 'pretrained':
                import tensorflow as tf
                obs_tf = {
                    k: tf.convert_to_tensor(np.expand_dims(v, axis=0), dtype=tf.float32)
                    for k, v in state['raw_observation'].items()
                }
                action = self.pretrained_policy(obs_tf).mean()[0].numpy()
            elif self.controller_type == 'connectome_hybrid':
                import tensorflow as tf
                obs_tf = {
                    k: tf.convert_to_tensor(np.expand_dims(v, axis=0), dtype=tf.float32)
                    for k, v in state['raw_observation'].items()
                }
                base_action = self.pretrained_policy(obs_tf).mean()[0].numpy()
                
                sensory = self.encoder.update(state)
                motor_rates = self.circuit.step(sensory)
                conn_action = self.decoder.decode(motor_rates, euler, state=state, target_z=target_z)
                motor_rates_history.append(motor_rates)
                
                action = base_action + self.connectome_blend * conn_action
                action = np.clip(action, self.env.env.action_spec().minimum,
                                 self.env.env.action_spec().maximum)
            else:
                sensory = self.encoder.update(state)
                motor_rates = self.circuit.step(sensory)
                action = self.decoder.decode(motor_rates, euler, state=state, target_z=target_z)
                motor_rates_history.append(motor_rates)

            actions.append(action.copy())

            # 3. Physics step
            state, reward, done = self.env.step(action)
            rewards.append(reward)

            if done:
                break

        # Compute summary metrics
        euler_arr = np.array(euler_angles) # (N, 3) in radians
        omega_arr = np.array(ang_velocities) # (N, 3) in rad/s
        pos_arr = np.array(positions)       # (N, 3) in meters
        act_arr = np.array(actions)         # (N, 12)

        rms_euler_deg = np.sqrt(np.mean(euler_arr**2, axis=0)) * (180.0 / np.pi)
        rms_total_deg = float(np.sqrt(np.mean(euler_arr**2))) * (180.0 / np.pi)
        
        peak_ang_vel = float(np.max(np.abs(omega_arr)))
        rms_ang_vel = float(np.sqrt(np.mean(omega_arr**2)))
        
        traj_drift = float(np.linalg.norm(pos_arr[-1] - pos_arr[0]))
        control_effort = float(np.mean(np.sum(act_arr**2, axis=1)))
        
        if recovery_time is None:
            recovery_time = duration - disturbance_time if disturbance_applied else 0.0

        return {
            'times': np.array(times),
            'positions': pos_arr,
            'euler_deg': euler_arr * (180.0 / np.pi),
            'ang_vel': omega_arr,
            'actions': act_arr,
            'rewards': np.array(rewards),
            'motor_rates': motor_rates_history,
            'rms_attitude_error_deg': rms_total_deg,
            'rms_roll_deg': float(rms_euler_deg[0]),
            'rms_pitch_deg': float(rms_euler_deg[1]),
            'rms_yaw_deg': float(rms_euler_deg[2]),
            'peak_ang_vel_rad_s': peak_ang_vel,
            'rms_ang_vel_rad_s': rms_ang_vel,
            'trajectory_drift_m': traj_drift,
            'control_effort': control_effort,
            'recovery_time_s': float(recovery_time),
            'crashed': crashed
        }
