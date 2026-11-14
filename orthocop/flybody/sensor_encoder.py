"""
Biophysical Sensor Encoder for Male Drosophila CNS Connectome.

Translates physical MuJoCo state measurements (angular velocity, orientation,
linear acceleration, wing kinematics) into neurobiological input currents
and firing rates for sensory afferents:
- Haltere Campaniform Sensilla (DMetaN afferents): 3-axis gyroscopic rate encoding
- Visual Lobula Plate Tangential Cells (LPTCs: VS, HS): Wide-field optic flow
- Wing Mechanoreceptors (ADMN, PDMN): Strain and stroke amplitude feedback
"""

from typing import Dict, Any, List
import numpy as np
from collections import deque


class BiophysicalSensorEncoder:
    """
    Encodes MuJoCo physical state measurements into sensory neuron inputs.
    Includes sensor time constants, neural transport latency delays, Gaussian noise,
    and physiological saturation limits.
    """

    def __init__(
        self,
        dt: float = 0.001,                   # Integration timestep (1.0 ms)
        haltere_tau_s: float = 0.001,        # 1.0 ms mechanical time constant
        haltere_delay_ms: float = 1.5,       # 1.5 ms axonal conduction delay
        haltere_noise_std: float = 0.05,     # Sensor measurement noise
        haltere_baseline_hz: float = 100.0,  # Resting spontaneous firing rate
        haltere_gain: float = 25.0,          # Hz per rad/s
        haltere_sat_hz: float = 600.0,       # Maximum physiological firing rate
        visual_tau: float = 0.015,           # 15 ms visual integration time constant
        visual_gain: float = 15.0,           # Hz per rad/s
        visual_sat_hz: float = 250.0         # Maximum visual LPTC firing rate
    ):
        self.dt = dt
        self.tau_s = haltere_tau_s
        self.noise_std = haltere_noise_std
        self.r0_haltere = haltere_baseline_hz
        self.gain_haltere = haltere_gain
        self.sat_haltere = haltere_sat_hz
        
        self.tau_v = visual_tau
        self.gain_vis = visual_gain
        self.sat_vis = visual_sat_hz

        # Filter states
        self.filtered_gyro = np.zeros(3)  # [p, q, r]
        self.filtered_vis = np.zeros(3)   # wide-field flow

        # Delay buffer for axonal transport delay
        self.delay_steps = max(1, int(round((haltere_delay_ms * 1e-3) / dt)))
        self.gyro_history = deque(maxlen=self.delay_steps + 1)
        for _ in range(self.delay_steps + 1):
            self.gyro_history.append(np.zeros(3))

    def update(self, state: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """
        Process new physical measurement from MuJoCo and compute sensory firing rates.
        
        Args:
            state: Dictionary containing MuJoCo state (ang_vel, quat, euler, etc.)
            
        Returns:
            Dict containing firing rates and injected currents for each sensory modality.
        """
        raw_gyro = state['ang_vel'].copy()  # [p, q, r] in rad/s
        
        # 1. First-order haltere mechanoreceptor dynamics: ds/dt = (-s + omega) / tau_s
        alpha_s = self.dt / (self.tau_s + self.dt)
        self.filtered_gyro = (1 - alpha_s) * self.filtered_gyro + alpha_s * raw_gyro
        
        # Add measurement noise
        noisy_gyro = self.filtered_gyro + np.random.normal(0, self.noise_std, size=3)
        self.gyro_history.append(noisy_gyro)
        
        # Extract delayed gyroscopic signal representing signal arrival at VNC
        delayed_gyro = self.gyro_history[0]
        p_del, q_del, r_del = delayed_gyro[0], delayed_gyro[1], delayed_gyro[2]

        # 2. First-order visual LPTC optical flow filter (VS / HS cells)
        alpha_v = self.dt / (self.tau_v + self.dt)
        self.filtered_vis = (1 - alpha_v) * self.filtered_vis + alpha_v * raw_gyro
        p_vis, q_vis, r_vis = self.filtered_vis[0], self.filtered_vis[1], self.filtered_vis[2]

        # 3. Bilateral Haltere Rate Encoding
        # Left haltere responds strongly to rightward roll (+p) and rightward yaw (+r)
        rate_haltere_L_roll = np.clip(self.r0_haltere + self.gain_haltere * (+p_del), 0.0, self.sat_haltere)
        rate_haltere_L_yaw  = np.clip(self.r0_haltere + self.gain_haltere * (+r_del), 0.0, self.sat_haltere)
        rate_haltere_L_pitch= np.clip(self.r0_haltere + self.gain_haltere * (+q_del), 0.0, self.sat_haltere)

        # Right haltere responds strongly to leftward roll (-p) and leftward yaw (-r)
        rate_haltere_R_roll = np.clip(self.r0_haltere + self.gain_haltere * (-p_del), 0.0, self.sat_haltere)
        rate_haltere_R_yaw  = np.clip(self.r0_haltere + self.gain_haltere * (-r_del), 0.0, self.sat_haltere)
        rate_haltere_R_pitch= np.clip(self.r0_haltere + self.gain_haltere * (+q_del), 0.0, self.sat_haltere)

        # Combined haltere sensory rate vectors
        rate_haltere_L = np.mean([rate_haltere_L_roll, rate_haltere_L_yaw, rate_haltere_L_pitch])
        rate_haltere_R = np.mean([rate_haltere_R_roll, rate_haltere_R_yaw, rate_haltere_R_pitch])

        # 4. Visual LPTC Rate Encoding (Horizontal HS cells encode yaw; Vertical VS encode roll/pitch)
        rate_hs_L = np.clip(50.0 + self.gain_vis * (+r_vis), 0.0, self.sat_vis)
        rate_hs_R = np.clip(50.0 + self.gain_vis * (-r_vis), 0.0, self.sat_vis)
        rate_vs_L = np.clip(50.0 + self.gain_vis * (+p_vis + q_vis), 0.0, self.sat_vis)
        rate_vs_R = np.clip(50.0 + self.gain_vis * (-p_vis + q_vis), 0.0, self.sat_vis)

        # 5. Wing Mechanoreceptor Encoding (strain from joint displacement)
        joints_pos = state.get('joints_pos', np.zeros(6))
        # Joints: yaw_L, roll_L, pitch_L, yaw_R, roll_R, pitch_R
        rate_wing_L = np.clip(100.0 + 30.0 * np.abs(joints_pos[0] if len(joints_pos) > 0 else 0.0), 0.0, 400.0)
        rate_wing_R = np.clip(100.0 + 30.0 * np.abs(joints_pos[3] if len(joints_pos) > 3 else 0.0), 0.0, 400.0)

        # Convert firing rates to effective injected depolarizing currents (nanoamperes)
        # 100 Hz baseline corresponds to ~0.4 nA maintaining subthreshold / tonic firing
        I_haltere_L = (rate_haltere_L / 100.0) * 0.45
        I_haltere_R = (rate_haltere_R / 100.0) * 0.45
        I_vis_L = (np.mean([rate_hs_L, rate_vs_L]) / 50.0) * 0.35
        I_vis_R = (np.mean([rate_hs_R, rate_vs_R]) / 50.0) * 0.35
        I_wing_L = (rate_wing_L / 100.0) * 0.30
        I_wing_R = (rate_wing_R / 100.0) * 0.30

        # 6. Leg Ground Contact (tactile bristles / hair plates) & Proprioception (FeCO)
        pos = state.get('pos', np.array([0.0, 0.0, 1.0]))
        altitude = pos[2] if len(pos) > 2 else 1.0
        # When approaching landing surface or ground (altitude < 0.25m), leg sensory bristles activate
        is_ground = max(0.0, min(1.0, (0.25 - altitude) / 0.15)) if altitude < 0.25 else 0.0
        contact_force = float(state.get('ground_contact', is_ground))
        I_leg_touch_L = contact_force * 0.50
        I_leg_touch_R = contact_force * 0.50
        I_leg_proprio_L = 0.35 + 0.15 * np.sin(p_del)
        I_leg_proprio_R = 0.35 - 0.15 * np.sin(p_del)

        # 7. Visual Looming Escape Trigger (sudden acceleration or looming stimulus)
        accel = state.get('lin_acc', np.zeros(3))
        loom_trigger = float(state.get('visual_loom', 1.0 if np.linalg.norm(accel) > 25.0 else 0.0))
        I_visual_loom = loom_trigger * 0.60

        return {
            'rate_haltere_L': rate_haltere_L,
            'rate_haltere_R': rate_haltere_R,
            'rate_vis_L': np.mean([rate_hs_L, rate_vs_L]),
            'rate_vis_R': np.mean([rate_hs_R, rate_vs_R]),
            'rate_wing_L': rate_wing_L,
            'rate_wing_R': rate_wing_R,
            'I_haltere_L': I_haltere_L,
            'I_haltere_R': I_haltere_R,
            'I_vis_L': I_vis_L,
            'I_vis_R': I_vis_R,
            'I_wing_L': I_wing_L,
            'I_wing_R': I_wing_R,
            'I_leg_touch_L': I_leg_touch_L,
            'I_leg_touch_R': I_leg_touch_R,
            'I_leg_proprio_L': I_leg_proprio_L,
            'I_leg_proprio_R': I_leg_proprio_R,
            'I_visual_loom': I_visual_loom,
            'p_del': p_del,
            'q_del': q_del,
            'r_del': r_del
        }
