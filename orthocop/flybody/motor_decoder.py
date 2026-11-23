"""
Motor Decoder: Male Drosophila CNS Connectome to MuJoCo FlyBody Actuators.

Converts population firing rates of extracted biological motor neurons
into calibrated 12-dimensional physical actuator commands for FlyBody:
  [0:3]   head: abduct, twist, pitch (gaze stabilization)
  [3:6]   left wing: yaw (stroke amp), roll (deviation), pitch (angle of attack)
  [6:9]   right wing: yaw, roll, pitch
  [9:11]  abdomen: abduct (yaw rudder), pitch (trim)
  [11]    user_0: wingbeat frequency modulation via WBPG

The WBPG (Wing Beat Pattern Generator) handles the base ~200 Hz flapping
pattern internally. The policy / connectome provides only SMALL steering
corrections (±0.3 typical, ±1.0 max) on top of the WBPG output.

Gains calibrated against pretrained RL policy action statistics:
  - Wing yaw:  mean ≈ ±0.07, std ≈ 0.20,  range [-0.85, +0.52]
  - Wing roll: mean ≈ +0.07, std ≈ 0.12,  range [-0.33, +0.50]
  - Wing pitch:mean ≈ -0.04, std ≈ 0.08,  range [-0.41, +0.23]
  - WBPG freq: mean ≈ -0.01, std ≈ 0.03,  range [-0.09, +0.03]
  - Head:      small, mean ≈ 0,   range [-0.35, +0.38]
  - Abdomen:   mean ≈ -0.01, std ≈ 0.09,  range [-0.31, +0.22]
"""

from typing import Dict, Any, Optional
import numpy as np


class MotorDecoder:
    """
    Decodes biological motor neuron firing rates into 12-DOF FlyBody actuator commands.

    Calibrated to match the pretrained RL policy's operating range.
    The WBPG handles base wingbeat; we provide only small corrections.
    """

    def __init__(
        self,
        # Fixed calibrated baselines (matched to 897-neuron circuit equilibrium rates)
        steer_baseline: float = 140.0,     # Steering MN resting rate (Hz)
        power_baseline: float = 135.0,     # Power MN (DLM/DVM) resting rate (Hz)
        # Actuator gains per Hz of bilateral difference
        # Calibrated so typical ±20 Hz firing rate deviation → ±0.1..0.3 action
        k_yaw: float = 0.008,             # Yaw actuator gain (stroke amplitude)
        k_roll: float = 0.005,            # Roll actuator gain (deviation / stroke tilt)
        k_pitch: float = 0.003,           # Wing pitch actuator gain (AoA)
        k_freq: float = 0.001,            # Flapping frequency gain (WBPG is internal)
        k_abd_yaw: float = 0.004,         # Abdominal rudder gain
        k_abd_pitch: float = 0.003,       # Abdominal pitch gain
        # PD attitude controller gains (haltere-equivalent reflex)
        # Calibrated to produce ±0.2..0.5 wing corrections for ±30° attitude errors
        kp_roll: float = 0.25,            # Proportional roll
        kd_roll: float = 0.08,            # Derivative roll damping (fast gust rejection)
        kp_pitch: float = 0.20,           # Proportional pitch
        kd_pitch: float = 0.08,           # Derivative pitch damping
        kp_yaw: float = 0.15,             # Proportional yaw
        kd_yaw: float = 0.05,             # Derivative yaw damping
        # Reference tracking (uses ref_displacement / ref_root_quat from obs)
        use_ref_tracking: bool = False,
        # Filter
        filter_tau: float = 0.003,         # Actuator smoothing (3 ms)
        dt: float = 0.0002                 # Control step (200 µs matching FlyBody)
    ):
        self.r0_steer = steer_baseline
        self.r0_power = power_baseline
        self.k_yaw = k_yaw
        self.k_roll = k_roll
        self.k_pitch = k_pitch
        self.k_freq = k_freq
        self.k_abd_yaw = k_abd_yaw
        self.k_abd_pitch = k_abd_pitch

        self.kp_roll = kp_roll
        self.kd_roll = kd_roll
        self.kp_pitch = kp_pitch
        self.kd_pitch = kd_pitch
        self.kp_yaw = kp_yaw
        self.kd_yaw = kd_yaw

        self.use_ref_tracking = use_ref_tracking
        self.dt = dt
        self.filter_tau = filter_tau
        self.prev_action = np.zeros(12)
        self._prev_euler = None

    def _quat_to_euler(self, q):
        """Convert quaternion [w,x,y,z] to euler [roll, pitch, yaw] in radians."""
        qw, qx, qy, qz = q[0], q[1], q[2], q[3]
        sinr_cosp = 2 * (qw * qx + qy * qz)
        cosr_cosp = 1 - 2 * (qx * qx + qy * qy)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2 * (qw * qy - qz * qx)
        pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
        siny_cosp = 2 * (qw * qz + qx * qy)
        cosy_cosp = 1 - 2 * (qy * qy + qz * qz)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    def _quat_error(self, q_current, q_target):
        """Compute attitude error quaternion (q_target * q_current^-1) as euler angles."""
        # q_current^-1 (conjugate for unit quaternion)
        q_inv = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
        # q_error = q_target * q_current^-1
        w1, x1, y1, z1 = q_target
        w2, x2, y2, z2 = q_inv
        q_err = np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        return self._quat_to_euler(q_err)

    def decode(
        self,
        motor_rates: Dict[str, float],
        euler_angles: Optional[np.ndarray] = None,
        state: Optional[Dict[str, Any]] = None,
        target_z: Optional[float] = None
    ) -> np.ndarray:
        """
        Compute FlyBody 12-dimensional actuator command from motor neuron firing rates.

        Args:
            motor_rates: Dictionary of average firing rates for motor neuron pools.
            euler_angles: Current body [roll, pitch, yaw] for PD stabilization.
            state: Full MuJoCo state dict.
            target_z: Optional target flight altitude (meters) for active biological altitude hold.

        Returns:
            np.ndarray of shape (12,)
        """
        action = np.zeros(12)
        r0 = self.r0_steer
        r0_pwr = self.r0_power

        # --- Extract angular velocity (derivative feedback) ---
        if euler_angles is not None and self._prev_euler is not None:
            d_euler = (euler_angles - self._prev_euler) / self.dt
            d_euler = np.clip(d_euler, -50.0, 50.0)
        else:
            d_euler = np.zeros(3)
        if euler_angles is not None:
            self._prev_euler = euler_angles.copy()

        # --- Use reference trajectory for attitude error if available ---
        euler_error = np.zeros(3)
        pos_error = np.zeros(3)
        if self.use_ref_tracking and state is not None:
            raw_obs = state.get('raw_observation', {})
            ref_quat = raw_obs.get('walker/ref_root_quat', None)
            ref_disp = raw_obs.get('walker/ref_displacement', None)
            if ref_quat is not None and len(ref_quat) > 0:
                # ref_root_quat[0] IS the relative quaternion from walker to ghost
                # Convert to euler to get the orientation error directly
                q_err = ref_quat[0]
                euler_error = self._quat_to_euler(q_err)
            if ref_disp is not None and len(ref_disp) > 0:
                # ref_displacement[0] = position of ghost relative to walker
                # Positive x = ghost is ahead, positive z = ghost is above
                pos_error = ref_disp[0]
        elif euler_angles is not None:
            # Fallback: track nominal flight trim (roll=0, pitch=-47.5 deg, yaw=0)
            euler_error = euler_angles.copy()
            euler_error[1] = euler_angles[1] - np.deg2rad(-47.5)

        # --- Use angular velocity from state if available (more accurate than finite diff) ---
        ang_vel = np.zeros(3)
        if state is not None:
            ang_vel = state.get('ang_vel', np.zeros(3))
        else:
            ang_vel = d_euler

        # --- Extract motor pool rates with fixed baselines ---
        roll_L = motor_rates.get('roll_L', r0)
        roll_R = motor_rates.get('roll_R', r0)
        pitch_L = motor_rates.get('pitch_L', r0)
        pitch_R = motor_rates.get('pitch_R', r0)
        yaw_L = motor_rates.get('yaw_L', r0)
        yaw_R = motor_rates.get('yaw_R', r0)

        dlm_L = motor_rates.get('DLM_L', r0_pwr)
        dlm_R = motor_rates.get('DLM_R', r0_pwr)
        dvm_L = motor_rates.get('DVM_L', r0_pwr)
        dvm_R = motor_rates.get('DVM_R', r0_pwr)

        abd_L = motor_rates.get('abd_L', r0)
        abd_R = motor_rates.get('abd_R', r0)
        abd_p = motor_rates.get('abd_pitch', r0)

        # --- Compute connectome-driven steering ---
        diff_yaw = (yaw_L - r0) - (yaw_R - r0)
        diff_roll = (roll_L - r0) - (roll_R - r0)
        diff_pitch = (pitch_L - r0) - (pitch_R - r0)

        conn_yaw = self.k_yaw * diff_yaw
        conn_roll = self.k_roll * diff_roll
        conn_pitch = self.k_pitch * diff_pitch

        # Power differential for frequency modulation
        dlm_mean = ((dlm_L - r0_pwr) + (dlm_R - r0_pwr)) / 2.0
        dvm_mean = ((dvm_L - r0_pwr) + (dvm_R - r0_pwr)) / 2.0

        # --- PD attitude controller (haltere-equivalent reflex) ---
        roll_err, pitch_err, yaw_err = euler_error[0], euler_error[1], euler_error[2]

        pd_roll = -self.kp_roll * roll_err - self.kd_roll * ang_vel[0]
        pd_pitch = -self.kp_pitch * pitch_err - self.kd_pitch * ang_vel[1]
        pd_yaw = -self.kp_yaw * yaw_err - self.kd_yaw * ang_vel[2]

        # --- Position error correction (track reference ghost position) ---
        # pos_error = [dx, dy, dz] from walker to ghost in world frame
        # Lateral (y): bank toward ghost → roll correction
        pos_roll_corr = 0.15 * pos_error[1] if len(pos_error) > 1 else 0.0
        # Vertical (z): ghost above → increase lift
        pos_lift_corr = 0.10 * pos_error[2] if len(pos_error) > 2 else 0.0
        # Forward (x): ghost ahead → pitch down to accelerate
        pos_pitch_corr = -0.10 * pos_error[0] if len(pos_error) > 0 else 0.0

        pd_roll += pos_roll_corr
        pd_pitch += pos_pitch_corr

        # --- 1. Head actuators [0:3]: Vestibulo-ocular reflex ---
        # Small corrections, range [-0.2, +0.2] / [-3.0, +3.0] / [-0.5, +0.3]
        action[0] = np.clip(-0.3 * yaw_err, -0.15, 0.15)
        action[1] = np.clip(-0.4 * roll_err, -0.5, 0.5)
        action[2] = np.clip(-0.3 * pitch_err, -0.1, 0.1)

        # --- 2. Left Wing [3:6] & Right Wing [6:9] ---
        # Connectome steering + PD correction
        # Wing yaw controls roll (bilateral differential → stroke amplitude)
        lw_yaw = conn_yaw + pd_roll
        rw_yaw = -conn_yaw - pd_roll

        # Wing roll controls pitch (bilateral symmetric → stroke plane tilt)
        lw_roll = conn_roll + pd_pitch
        rw_roll = conn_roll + pd_pitch  # Symmetric for pitch

        # Wing pitch (angle of attack)
        lw_pitch = conn_pitch
        rw_pitch = -conn_pitch

        # Clip to action spec range (allow full range for large corrections)
        action[3] = np.clip(lw_yaw, -1.0, 1.0)
        action[4] = np.clip(lw_roll, -1.0, 1.0)
        action[5] = np.clip(lw_pitch, -1.0, 1.0)
        action[6] = np.clip(rw_yaw, -1.0, 1.0)
        action[7] = np.clip(rw_roll, -1.0, 1.0)
        action[8] = np.clip(rw_pitch, -1.0, 1.0)

        # --- 3. Abdomen [9:11]: Inertial rudder + trim ---
        abd_yaw_cmd = self.k_abd_yaw * ((abd_L - r0) - (abd_R - r0)) + pd_yaw * 0.5
        abd_pitch_cmd = -self.k_abd_pitch * (abd_p - r0) + pd_pitch * 0.3
        action[9] = np.clip(abd_yaw_cmd, -0.7, 0.7)
        action[10] = np.clip(abd_pitch_cmd, -1.05, 0.7)

        # --- 4. Flapping Frequency (WBPG) [11]: Active Biological Altitude Hold & Power ---
        # The WBPG handles wing flapping internally at ~200 Hz
        delta_freq = self.k_freq * (dlm_mean - dvm_mean) + pos_lift_corr
        if target_z is not None and state is not None:
            alt_err = target_z - state['pos'][2]
            vz = state['lin_vel'][2]
            cos_roll = max(0.5, np.cos(state['euler'][0]))
            bank_comp = (1.0 / cos_roll - 1.0) * 0.20
            delta_freq += 1.4 * alt_err - 0.012 * vz + 0.16 + bank_comp
        action[11] = np.clip(delta_freq, -1.0, 1.0)

        # --- Smooth action transitions via 1st-order low-pass filter ---
        alpha = self.dt / (self.filter_tau + self.dt)
        smoothed = (1.0 - alpha) * self.prev_action + alpha * action
        self.prev_action = smoothed.copy()

        return smoothed
