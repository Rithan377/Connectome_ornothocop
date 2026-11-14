"""
Closed-Loop Flight Simulation Engine.
Couples:
Environment -> Aerodynamics -> 6-DOF Rigid-Body Dynamics -> Sensory Suite -> Brian2 Connectome Neural Circuit -> Wing Kinematics/Actuator -> Flight Dynamics.
Calculates Flight Stability Metrics (Orientation error, Recovery Time, Peak Error, RMS Error, Trajectory Drift, Failure Probability).
"""

import numpy as np
import pandas as pd
import brian2 as b2
from orthocop.simulations.physics_model import DrosophilaFlightDynamics, quat_to_rot_matrix, rot_matrix_to_euler
from orthocop.simulations.environment import FlightEnvironment
from orthocop.simulations.sensors import DrosophilaSensorySuite
from orthocop.simulations.neural_model import ConnectomeFlightNeuralModel

class ClosedLoopFlightSimulator:
    def __init__(self, neural_model=None, nodes_df=None, edges_df=None, ablated_neurons=None, dt=1e-4):
        self.dt = dt
        self.physics = DrosophilaFlightDynamics()
        self.sensors = DrosophilaSensorySuite(dt=dt)
        
        if neural_model is not None:
            self.neural_model = neural_model
        else:
            if nodes_df is None:
                nodes_df = pd.read_feather("circuits/flight_candidate_neurons.feather")
            if edges_df is None:
                edges_df = pd.read_feather("circuits/flight_subnetwork_edges_min2.feather")
            self.neural_model = ConnectomeFlightNeuralModel(nodes_df, edges_df, ablated_neurons=ablated_neurons)

    def run_simulation(
        self,
        duration_s=0.15,
        disturbance_type='roll_kick', # 'none', 'roll_kick', 'pitch_kick', 'gust', 'turbulence'
        disturbance_mag=8.0, # rad/s or m/s
        disturbance_time=0.03, # 30 ms
        u_wind_base=np.zeros(3)
    ):
        """
        Executes a closed-loop flight simulation run.
        Returns detailed trajectory dataframe, performance metrics, and cost J.
        """
        dt = self.dt
        total_steps = int(round(duration_s / dt))
        
        # Initial flight state (equilibrium hover at z = 0.5 m)
        state = {
            'p': np.array([0.0, 0.0, 0.5]),
            'v': np.array([0.0, 0.0, 0.0]),
            'q': np.array([1.0, 0.0, 0.0, 0.0]),
            'omega': np.array([0.0, 0.0, 0.0]),
            't': 0.0
        }
        
        # Setup environment
        env = FlightEnvironment(u_0=u_wind_base, dt=dt)
        if disturbance_type == 'gust':
            env.add_gaussian_gust(amplitude=[0.0, disturbance_mag, 0.0], t_peak=disturbance_time, sigma_duration=0.008)
            
        trajectory = []
        wing_ctrl = {'d_A_L': 0.0, 'd_A_R': 0.0, 'd_phi_0_L': 0.0, 'd_phi_0_R': 0.0}
        
        failed = False
        t_stable = None
        stable_window_count = 0
        min_stable_steps = int(round(0.015 / dt)) # 15 ms stable confirmation window
        
        # Step closed-loop in sync
        # Neural update window: update wing commands every 2.0 ms (approx every half wingbeat)
        neural_dt_steps = int(round(0.002 / dt))
        
        for step in range(total_steps):
            t_curr = state['t']
            
            # Apply impulsive disturbance torque kick at disturbance_time if configured
            if disturbance_type == 'roll_kick' and abs(t_curr - disturbance_time) < dt/2:
                state['omega'][0] += disturbance_mag
            elif disturbance_type == 'pitch_kick' and abs(t_curr - disturbance_time) < dt/2:
                state['omega'][1] += disturbance_mag
            elif disturbance_type == 'yaw_kick' and abs(t_curr - disturbance_time) < dt/2:
                state['omega'][2] += disturbance_mag
                
            # 1. Environment wind
            wind_world = env.step(t_curr)
            
            # 2. Physics integration step
            state = self.physics.step(state, wing_ctrl, wind_world, dt)
            
            # Check failure condition (tumble > 85 deg or crash z <= 0)
            R_curr = quat_to_rot_matrix(state['q'])
            # Desired orientation is level flight (Identity rotation matrix)
            R_e = R_curr # R_d^T * R = I * R = R
            cos_theta = np.clip((np.trace(R_e) - 1.0) / 2.0, -1.0, 1.0)
            theta_e_rad = np.arccos(cos_theta)
            theta_e_deg = np.rad2deg(theta_e_rad)
            
            if theta_e_deg > 85.0 or state['p'][2] <= 0.0:
                failed = True
                
            # Check stability recovery condition after disturbance
            if t_curr > disturbance_time and not failed:
                if theta_e_deg <= 5.0 and np.linalg.norm(state['omega']) <= 2.0:
                    stable_window_count += 1
                    if stable_window_count >= min_stable_steps and t_stable is None:
                        t_stable = t_curr - (min_stable_steps * dt)
                else:
                    stable_window_count = 0
                    
            # 3. Sensor measurement
            aerodynamics = {'F_z': state['F_aero'][2]}
            sensory_readings = self.sensors.process(state, aerodynamics)
            
            # 4. Neural Circuit update every neural_dt_steps (2 ms)
            if step % neural_dt_steps == 0:
                self.neural_model.set_sensory_inputs(sensory_readings)
                self.neural_model.step(2.0 * b2.ms)
                wing_ctrl = self.neural_model.decode_motor_commands(window_ms=5.0)
                
            trajectory.append({
                't': t_curr,
                'x': state['p'][0],
                'y': state['p'][1],
                'z': state['p'][2],
                'vx': state['v'][0],
                'vy': state['v'][1],
                'vz': state['v'][2],
                'roll_deg': state['euler_deg'][0],
                'pitch_deg': state['euler_deg'][1],
                'yaw_deg': state['euler_deg'][2],
                'omega_x': state['omega'][0],
                'omega_y': state['omega'][1],
                'omega_z': state['omega'][2],
                'theta_e_deg': theta_e_deg,
                'ctrl_dA_L': wing_ctrl['d_A_L'],
                'ctrl_dA_R': wing_ctrl['d_A_R']
            })
            
            if failed:
                break
                
        df_traj = pd.DataFrame(trajectory)
        
        # Calculate quantitative stability metrics
        e_rms = np.sqrt(np.mean(df_traj['theta_e_deg']**2)) if len(df_traj) > 0 else 90.0
        e_peak = np.max(df_traj['theta_e_deg']) if len(df_traj) > 0 else 90.0
        
        if t_stable is not None:
            t_recovery = max(0.0, t_stable - disturbance_time)
        else:
            t_recovery = duration_s - disturbance_time # Max penalty if unrecovered
            
        traj_drift = np.sqrt((df_traj['x'].iloc[-1])**2 + (df_traj['y'].iloc[-1])**2) if len(df_traj) > 0 else 1.0
        
        # Cost function: J = lambda1 * E_orient + lambda2 * T_recovery + lambda3 * E_traj + lambda4 * P_fail
        lambda_orient = 1.0
        lambda_recov = 2.0
        lambda_drift = 0.5
        lambda_fail = 50.0
        
        fail_val = 1.0 if failed else 0.0
        J_cost = (
            lambda_orient * (e_rms / 10.0) +
            lambda_recov * (t_recovery / 0.05) +
            lambda_drift * (traj_drift / 0.05) +
            lambda_fail * fail_val
        )
        
        metrics = {
            'failed': failed,
            'E_rms_deg': float(e_rms),
            'E_peak_deg': float(e_peak),
            'T_recovery_s': float(t_recovery),
            'trajectory_drift_m': float(traj_drift),
            'J_cost': float(J_cost),
            'duration_completed_s': float(df_traj['t'].iloc[-1]) if len(df_traj) > 0 else 0.0
        }
        
        return df_traj, metrics

if __name__ == "__main__":
    nodes = pd.read_feather("circuits/flight_candidate_neurons.feather")
    edges = pd.read_feather("circuits/flight_subnetwork_edges_min2.feather")
    
    print("[*] Running Closed-Loop Flight Simulation Test...")
    sim = ClosedLoopFlightSimulator(nodes_df=nodes, edges_df=edges)
    
    # Run roll disturbance rejection
    df_traj, metrics = sim.run_simulation(
        duration_s=0.10,
        disturbance_type='roll_kick',
        disturbance_mag=10.0, # 10 rad/s roll impulse kick
        disturbance_time=0.02
    )
    
    print("\nSimulation Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"\nFinal orientation error: {df_traj['theta_e_deg'].iloc[-1]:.2f} deg")
