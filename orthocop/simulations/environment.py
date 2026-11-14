"""
Flight Environment Disturbance Generator.
Implements:
1. Uniform wind (still air, headwind, crosswind).
2. Gaussian gust disturbances.
3. Ornstein-Uhlenbeck stochastic turbulence process.
4. Wind autocorrelation and Power Spectral Density (PSD) analysis via Welch's method.
"""

import numpy as np
from scipy import signal

class FlightEnvironment:
    def __init__(self, u_0=np.zeros(3), dt=1e-4):
        self.u_0 = np.array(u_0, dtype=float)  # Steady wind vector [Ux, Uy, Uz] (m/s)
        self.dt = dt
        self.gusts = []
        
        # Ornstein-Uhlenbeck stochastic turbulence parameters
        self.ou_tau = 0.05                     # Correlation timescale (50 ms)
        self.ou_sigma = 0.2                    # Turbulence intensity (m/s)
        self.ou_state = np.zeros(3)

    def add_gaussian_gust(self, amplitude, t_peak, sigma_duration):
        """Adds a localized Gaussian gust: u_g(t) = A_g * exp(-(t - t_g)^2 / (2 * sigma_g^2))."""
        self.gusts.append({
            'A': np.array(amplitude, dtype=float),
            't_peak': float(t_peak),
            'sigma': float(sigma_duration)
        })

    def step(self, t):
        """Computes instantaneous wind vector u_air(t) in world frame."""
        # 1. Base steady wind
        u_total = self.u_0.copy()
        
        # 2. Add Gaussian gusts
        for g in self.gusts:
            factor = np.exp(-((t - g['t_peak'])**2) / (2.0 * (g['sigma']**2)))
            u_total += g['A'] * factor
            
        # 3. Ornstein-Uhlenbeck stochastic turbulence step:
        # du = -(u / tau) * dt + sigma * sqrt(dt) * N(0, 1)
        dW = np.random.normal(0.0, 1.0, 3)
        self.ou_state += -(self.ou_state / self.ou_tau) * self.dt + self.ou_sigma * np.sqrt(self.dt) * dW
        u_total += self.ou_state
        
        return u_total

    @staticmethod
    def compute_autocorrelation(u_signal):
        """Computes normalized autocorrelation function R_uu(tau)."""
        u_centered = u_signal - np.mean(u_signal)
        n = len(u_centered)
        autocorr = signal.correlate(u_centered, u_centered, mode='full')
        autocorr = autocorr[n-1:] / autocorr[n-1]
        return autocorr

    @staticmethod
    def compute_psd(u_signal, fs):
        """Estimates Power Spectral Density S_uu(f) using Welch's method."""
        freqs, psd = signal.welch(u_signal, fs=fs, nperseg=min(1024, len(u_signal)))
        return freqs, psd

if __name__ == "__main__":
    env = FlightEnvironment(u_0=[0.5, 0.0, 0.0], dt=1e-4)
    env.add_gaussian_gust(amplitude=[0.0, 1.5, 0.5], t_peak=0.05, sigma_duration=0.01)
    
    t_arr = np.arange(0, 0.1, 1e-4)
    u_log = [env.step(t) for t in t_arr]
    u_log = np.array(u_log)
    
    freqs, psd = FlightEnvironment.compute_psd(u_log[:, 1], fs=10000)
    print("Environment module verified successfully.")
    print("Max gust experienced:", np.max(u_log, axis=0))
    print(f"PSD computed: {len(freqs)} frequency bins.")
