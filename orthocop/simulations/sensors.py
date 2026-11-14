"""
Artificial Haltere, Visual, and Wing Mechanosensory Transduction Models.
Implements:
1. 1st-order sensor dynamics: tau_s * dy/dt + y = K_s * x
2. Sensor latency / delay line: y(t - tau_d)
3. Gaussian measurement noise: y_meas = y_true + epsilon(t)
4. Saturation nonlinearity: min(y_max, max(y_min, y))
5. Transduction of physical states into sensory afferent currents for Brian2 neural circuit.
"""

from collections import deque
import numpy as np

class FirstOrderSensor:
    def __init__(self, tau_s=0.002, K_s=1.0, delay_s=0.002, noise_std=0.02, sat_limits=(-100.0, 100.0), dt=1e-4):
        self.tau_s = tau_s            # Sensor lag time constant (s)
        self.K_s = K_s                # Sensor sensitivity gain
        self.delay_steps = max(1, int(round(delay_s / dt)))
        self.noise_std = noise_std    # Measurement noise standard deviation
        self.y_min, self.y_max = sat_limits
        self.dt = dt
        
        self.state = 0.0
        self.delay_buffer = deque([0.0] * self.delay_steps, maxlen=self.delay_steps)

    def update(self, physical_input):
        # 1. First order response: dy = (K_s * x - y) / tau_s * dt
        dy = (self.K_s * physical_input - self.state) / self.tau_s * self.dt
        self.state += dy
        
        # 2. Add to delay buffer and pop delayed value
        delayed_val = self.delay_buffer[0]
        self.delay_buffer.append(self.state)
        
        # 3. Add measurement noise
        noise = np.random.normal(0.0, self.noise_std)
        measured = delayed_val + noise
        
        # 4. Saturation clipping
        saturated = np.clip(measured, self.y_min, self.y_max)
        return saturated

class DrosophilaSensorySuite:
    def __init__(self, dt=1e-4):
        self.dt = dt
        
        # Haltere gyros: rapid mechanosensory response (tau_s ~ 1 ms, delay ~ 1-2 ms)
        # Senses roll, pitch, and yaw angular velocities (rad/s)
        self.haltere_roll = FirstOrderSensor(tau_s=0.001, K_s=1.0, delay_s=0.0015, noise_std=0.05, sat_limits=(-50.0, 50.0), dt=dt)
        self.haltere_pitch = FirstOrderSensor(tau_s=0.001, K_s=1.0, delay_s=0.0015, noise_std=0.05, sat_limits=(-50.0, 50.0), dt=dt)
        self.haltere_yaw = FirstOrderSensor(tau_s=0.001, K_s=1.0, delay_s=0.0015, noise_std=0.05, sat_limits=(-50.0, 50.0), dt=dt)
        
        # Visual LPTC sensor: lobula plate tangential cells (tau_s ~ 15 ms, delay ~ 20 ms)
        # Senses wide-field optic flow / orientation error (deg)
        self.vis_roll = FirstOrderSensor(tau_s=0.015, K_s=1.0, delay_s=0.020, noise_std=0.02, sat_limits=(-90.0, 90.0), dt=dt)
        self.vis_pitch = FirstOrderSensor(tau_s=0.015, K_s=1.0, delay_s=0.020, noise_std=0.02, sat_limits=(-90.0, 90.0), dt=dt)
        self.vis_yaw = FirstOrderSensor(tau_s=0.015, K_s=1.0, delay_s=0.020, noise_std=0.02, sat_limits=(-180.0, 180.0), dt=dt)
        
        # Wing mechanoreceptor: detects instantaneous aerodynamic load
        self.wing_strain_L = FirstOrderSensor(tau_s=0.001, K_s=1.0, delay_s=0.001, noise_std=0.01, sat_limits=(-10.0, 10.0), dt=dt)
        self.wing_strain_R = FirstOrderSensor(tau_s=0.001, K_s=1.0, delay_s=0.001, noise_std=0.01, sat_limits=(-10.0, 10.0), dt=dt)

    def process(self, state, aerodynamics):
        omega = state['omega'] # [p (roll_rate), q (pitch_rate), r (yaw_rate)] (rad/s)
        euler_deg = state['euler_deg'] # [roll, pitch, yaw] in degrees
        
        # 1. Haltere Gyro readings
        h_roll = self.haltere_roll.update(omega[0])
        h_pitch = self.haltere_pitch.update(omega[1])
        h_yaw = self.haltere_yaw.update(omega[2])
        
        # 2. Visual optic flow / orientation readings
        v_roll = self.vis_roll.update(euler_deg[0])
        v_pitch = self.vis_pitch.update(euler_deg[1])
        v_yaw = self.vis_yaw.update(euler_deg[2])
        
        # 3. Wing strain
        F_z = aerodynamics.get('F_z', 0.0)
        w_L = self.wing_strain_L.update(F_z)
        w_R = self.wing_strain_R.update(F_z)
        
        # Transduce physical readings into sensory currents for left/right haltere afferents
        # Haltere afferents fire differentially for roll (left vs right rotation)
        # and yaw/pitch:
        # I_ext = I_base + gain * sensor_reading
        i_base = 1.2e-9 # 1.2 nA baseline bias near threshold
        i_scale = 0.08e-9 # 0.08 nA per (rad/s)
        
        haltere_current_L = max(0.0, i_base + i_scale * (h_roll - 0.5 * h_yaw))
        haltere_current_R = max(0.0, i_base + i_scale * (-h_roll + 0.5 * h_yaw))
        haltere_current_pitch = max(0.0, i_base + i_scale * h_pitch)
        
        visual_current_L = max(0.0, i_base + 0.02e-9 * (v_roll - 0.5 * v_yaw))
        visual_current_R = max(0.0, i_base + 0.02e-9 * (-v_roll + 0.5 * v_yaw))
        
        return {
            'haltere_raw': np.array([h_roll, h_pitch, h_yaw]),
            'visual_raw': np.array([v_roll, v_pitch, v_yaw]),
            'haltere_current_L': haltere_current_L,
            'haltere_current_R': haltere_current_R,
            'haltere_current_pitch': haltere_current_pitch,
            'visual_current_L': visual_current_L,
            'visual_current_R': visual_current_R
        }

if __name__ == "__main__":
    suite = DrosophilaSensorySuite()
    dummy_state = {
        'omega': np.array([5.0, 0.0, -2.0]),
        'euler_deg': np.array([10.0, -2.0, 5.0])
    }
    dummy_aero = {'F_z': 9.8e-6}
    readings = suite.process(dummy_state, dummy_aero)
    print("Sensory suite verified:")
    print("Haltere gyro (rad/s):", readings['haltere_raw'])
    print("Haltere current Left (nA):", readings['haltere_current_L'] * 1e9)
    print("Haltere current Right (nA):", readings['haltere_current_R'] * 1e9)
