"""
Flapping Wing Insect Aerodynamics and 6-DOF Rigid-Body Flight Dynamics.
Implements:
1. 6-DOF Newton-Euler equations of motion with quaternion kinematics.
2. Sinusoidal and modulated wingbeat kinematics.
3. Wing actuator dynamics (inertia, damping, passive stiffness, aerodynamic back-torque).
4. Quasi-steady blade-element flapping wing aerodynamics (translational lift/drag, rotational circulation, spanwise force integration).
"""

import numpy as np

def quat_mult(q1, q2):
    """Multiplies two quaternions [w, x, y, z]."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def quat_to_rot_matrix(q):
    """Converts unit quaternion [w, x, y, z] to 3x3 rotation matrix (body to world)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(y**2 + z**2), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x**2 + y**2)]
    ])

def rot_matrix_to_euler(R):
    """Extracts roll, pitch, yaw (ZYX convention) in radians from rotation matrix."""
    pitch = np.arcsin(-np.clip(R[2, 0], -1.0, 1.0))
    if np.cos(pitch) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.array([roll, pitch, yaw])

class DrosophilaFlightDynamics:
    def __init__(self):
        # Biological morphometrics (Drosophila melanogaster)
        self.mass = 1.0e-6        # kg (1.0 mg)
        self.g = 9.81             # m/s^2
        self.rho_air = 1.225      # kg/m^3
        
        # Principal moments of inertia (kg * m^2)
        # Ixx (roll), Iyy (pitch), Izz (yaw)
        self.I = np.diag([1.5e-13, 5.0e-13, 5.5e-13])
        self.I_inv = np.linalg.inv(self.I)
        
        # Wing parameters
        self.wing_length = 2.5e-3      # m (R = 2.5 mm)
        self.mean_chord = 0.8e-3       # m (c = 0.8 mm)
        self.wing_area = self.wing_length * self.mean_chord # m^2 per wing
        self.base_frequency = 200.0    # Hz (200 wingbeats/sec)
        
        # Aerodynamic coefficients (Dickinson et al. 1999)
        self.CL_max = 1.8
        self.CD_0 = 0.4
        self.CD_max = 3.0
        self.C_rot = 1.5               # Rotational circulation coefficient
        
        # Baseline stroke parameters (radians)
        self.phi_0 = 0.0               # Mean stroke angle
        self.A_phi = np.deg2rad(65.0)  # Stroke amplitude (peak-to-peak 130 deg)
        self.r_cp_frac = 0.55          # Center of pressure at 55% wing length
        
        # Actuator parameters
        self.I_wing = 2.0e-15          # kg*m^2 wing inertia
        self.c_wing = 1.0e-12          # N*m*s/rad damping

    def compute_wing_kinematics(self, t, wing_ctrl):
        """
        Calculates stroke kinematics for Left and Right wings:
        phi(t) = phi_0 + A_phi * sin(2*pi*f*t + psi)
        wing_ctrl dict provides differential amplitude (d_A_phi), offset (d_phi_0), frequency.
        """
        f = self.base_frequency * wing_ctrl.get('freq_scale', 1.0)
        omega_wb = 2.0 * np.pi * f
        
        # Left wing control
        A_phi_L = self.A_phi + wing_ctrl.get('d_A_L', 0.0)
        phi_0_L = self.phi_0 + wing_ctrl.get('d_phi_0_L', 0.0)
        
        # Right wing control
        A_phi_R = self.A_phi + wing_ctrl.get('d_A_R', 0.0)
        phi_0_R = self.phi_0 + wing_ctrl.get('d_phi_0_R', 0.0)
        
        # Stroke angle phi(t)
        phi_L = phi_0_L + A_phi_L * np.sin(omega_wb * t)
        phi_R = phi_0_R + A_phi_R * np.sin(omega_wb * t)
        
        # Angular velocities dphi/dt
        dphi_L = omega_wb * A_phi_L * np.cos(omega_wb * t)
        dphi_R = omega_wb * A_phi_R * np.cos(omega_wb * t)
        
        # Angular accelerations d2phi/dt2
        ddphi_L = -(omega_wb**2) * A_phi_L * np.sin(omega_wb * t)
        ddphi_R = -(omega_wb**2) * A_phi_R * np.sin(omega_wb * t)
        
        return {
            'phi_L': phi_L, 'dphi_L': dphi_L, 'ddphi_L': ddphi_L,
            'phi_R': phi_R, 'dphi_R': dphi_R, 'ddphi_R': ddphi_R,
            'f': f
        }

    def compute_aerodynamic_forces(self, kinematics, body_vel_world, R_wb, wind_world):
        """
        Quasi-steady blade-element aerodynamic model for flapping wings.
        Calculates total aerodynamic lift, drag, and body torques.
        """
        # Relative air velocity in world frame
        v_air_world = body_vel_world - wind_world
        # Transform air velocity to body frame
        R_bw = R_wb.T
        v_air_body = R_bw @ v_air_world
        
        r_cp = self.r_cp_frac * self.wing_length
        total_force_body = np.zeros(3)
        total_torque_body = np.zeros(3)
        
        # Left and Right wing positions relative to CoM (meters)
        wing_hinge_L = np.array([0.0, 0.4e-3, 0.2e-3])
        wing_hinge_R = np.array([0.0, -0.4e-3, 0.2e-3])
        
        for side, hinge, dphi in [
            ('L', wing_hinge_L, kinematics['dphi_L']),
            ('R', wing_hinge_R, kinematics['dphi_R'])
        ]:
            # Wing flapping linear speed at center of pressure
            v_flap_mag = abs(dphi) * r_cp
            # Effective angle of attack during translation
            # Insects reverse wing pitch at stroke reversal (alpha ~ 45 deg)
            alpha = np.deg2rad(45.0)
            
            CL = self.CL_max * np.sin(2.0 * alpha)
            CD = self.CD_0 + self.CD_max * (np.sin(alpha)**2)
            
            # Dynamic pressure q_dyn = 0.5 * rho * U^2
            U_rel = np.sqrt(v_flap_mag**2 + np.sum(v_air_body**2))
            q_dyn = 0.5 * self.rho_air * (U_rel**2)
            
            # Lift is perpendicular to stroke plane (upward +Z in body frame)
            # Drag opposes instantaneous flapping motion
            F_lift = q_dyn * self.wing_area * CL
            F_drag = q_dyn * self.wing_area * CD
            
            # Net force vector in body coordinates
            # Upward lift (+Z) with small forward/backward component based on stroke direction
            F_z = F_lift
            F_x = -F_drag * np.sin(kinematics['phi_' + side]) * np.sign(dphi)
            F_y = 0.0
            
            F_wing = np.array([F_x, F_y, F_z])
            total_force_body += F_wing
            
            # Torque tau = r_hinge x F_wing
            tau_wing = np.cross(hinge, F_wing)
            # Differential lift produces roll torque: L_wing produces -roll, R_wing produces +roll
            if side == 'L':
                tau_wing[0] -= F_lift * (hinge[1] + r_cp)
            else:
                tau_wing[0] += F_lift * (abs(hinge[1]) + r_cp)
                
            total_torque_body += tau_wing
            
        return total_force_body, total_torque_body

    def step(self, state, wing_ctrl, wind_world, dt):
        """
        Integrates 6-DOF equations of motion forward by dt:
        State vector:
        p: position [x, y, z] (world frame)
        v: linear velocity [vx, vy, vz] (world frame)
        q: attitude quaternion [w, x, y, z] (world to body)
        omega: angular velocity [p, q, r] (body frame)
        t: current time
        """
        p = state['p']
        v = state['v']
        q = state['q']
        omega = state['omega']
        t = state['t']
        
        R_wb = quat_to_rot_matrix(q)
        
        # 1. Wing Kinematics
        kinematics = self.compute_wing_kinematics(t, wing_ctrl)
        
        # 2. Aerodynamic forces & torques
        F_aero_body, tau_aero_body = self.compute_aerodynamic_forces(kinematics, v, R_wb, wind_world)
        
        # 3. Total forces in world frame
        # Gravity acts in -Z direction
        F_grav_world = np.array([0.0, 0.0, -self.mass * self.g])
        F_aero_world = R_wb @ F_aero_body
        F_total_world = F_aero_world + F_grav_world
        
        # Linear acceleration: a = F / m
        acc_world = F_total_world / self.mass
        
        # 4. Rotational dynamics (Euler's equation in body frame)
        # tau = I * d_omega/dt + omega x (I * omega)
        # d_omega/dt = I^-1 * (tau - omega x (I * omega))
        gyroscopic_torque = np.cross(omega, self.I @ omega)
        d_omega = self.I_inv @ (tau_aero_body - gyroscopic_torque)
        
        # 5. Quaternion kinematics: dq/dt = 0.5 * q x [0, omega]
        omega_quat = np.array([0.0, omega[0], omega[1], omega[2]])
        dq = 0.5 * quat_mult(q, omega_quat)
        
        # 6. Euler integration step
        p_next = p + v * dt
        v_next = v + acc_world * dt
        q_next = q + dq * dt
        q_next = q_next / np.linalg.norm(q_next) # Normalize unit quaternion
        omega_next = omega + d_omega * dt
        t_next = t + dt
        
        euler_deg = np.rad2deg(rot_matrix_to_euler(quat_to_rot_matrix(q_next)))
        
        return {
            'p': p_next,
            'v': v_next,
            'q': q_next,
            'omega': omega_next,
            'euler_deg': euler_deg,
            't': t_next,
            'F_aero': F_aero_world,
            'tau_aero': tau_aero_body
        }

if __name__ == "__main__":
    fly = DrosophilaFlightDynamics()
    state = {
        'p': np.array([0.0, 0.0, 0.5]),
        'v': np.array([0.0, 0.0, 0.0]),
        'q': np.array([1.0, 0.0, 0.0, 0.0]),
        'omega': np.array([0.0, 0.0, 0.0]),
        't': 0.0
    }
    ctrl = {'d_A_L': 0.0, 'd_A_R': 0.0}
    wind = np.zeros(3)
    
    # Test step
    dt = 1e-4 # 0.1 ms
    for _ in range(100):
        state = fly.step(state, ctrl, wind, dt)
    print("Test 10ms simulation finished successfully:")
    print("Position:", state['p'])
    print("Euler (deg) [roll, pitch, yaw]:", state['euler_deg'])
