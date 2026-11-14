"""
Monte Carlo Flight Simulation and Statistical Uncertainty Quantification.
Runs multiple stochastic trials varying gust amplitudes, sensor noise, and initial body orientations.
Computes sample mean, variance, standard deviation, 95% Student's t confidence interval, percentiles, and failure probability.
"""

import os
import numpy as np
import pandas as pd
from scipy import stats
from orthocop.simulations.closed_loop import ClosedLoopFlightSimulator

def run_monte_carlo_experiments(n_trials=40, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    nodes = pd.read_feather("circuits/flight_candidate_neurons.feather")
    edges = pd.read_feather("circuits/flight_subnetwork_edges_min2.feather")
    
    sim = ClosedLoopFlightSimulator(nodes_df=nodes, edges_df=edges)
    
    print(f"[*] Starting Monte Carlo evaluation across {n_trials} randomized disturbance trials...")
    trial_records = []
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    for i in range(n_trials):
        # Vary disturbance magnitude: 5 to 15 rad/s
        roll_mag = np.random.uniform(5.0, 15.0) * np.random.choice([-1.0, 1.0])
        pitch_mag = np.random.uniform(0.0, 5.0) * np.random.choice([-1.0, 1.0])
        
        # Vary crosswind: -1.0 to 1.0 m/s
        wind_y = np.random.uniform(-1.0, 1.0)
        u_wind = np.array([0.0, wind_y, 0.0])
        
        df_traj, metrics = sim.run_simulation(
            duration_s=0.08,
            disturbance_type='roll_kick',
            disturbance_mag=roll_mag,
            disturbance_time=0.015,
            u_wind_base=u_wind
        )
        
        metrics['trial_id'] = i
        metrics['roll_kick_mag'] = float(roll_mag)
        metrics['crosswind_mag'] = float(wind_y)
        trial_records.append(metrics)
        
        if (i + 1) % 10 == 0 or i == n_trials - 1:
            print(f"    Completed {i + 1}/{n_trials} trials... (Latest J: {metrics['J_cost']:.2f})")
            
    df_trials = pd.DataFrame(trial_records)
    out_csv = os.path.join(output_dir, "monte_carlo_trials.csv")
    df_trials.to_csv(out_csv, index=False)
    
    # Statistical analysis
    costs = df_trials['J_cost'].values
    N = len(costs)
    mean_J = float(np.mean(costs))
    var_J = float(np.var(costs, ddof=1))
    std_J = float(np.std(costs, ddof=1))
    
    # 95% Confidence Interval: mean +/- t * std / sqrt(N)
    alpha = 0.05
    t_crit = float(stats.t.ppf(1.0 - alpha/2.0, df=N - 1))
    ci_margin = float(t_crit * (std_J / np.sqrt(N)))
    ci_lower = mean_J - ci_margin
    ci_upper = mean_J + ci_margin
    
    p_failure = float(np.mean(df_trials['failed']))
    
    stats_summary = {
        "n_trials": N,
        "mean_J": mean_J,
        "variance_J": var_J,
        "std_J": std_J,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "ci_margin": ci_margin,
        "percentile_25": float(np.percentile(costs, 25)),
        "median_J": float(np.median(costs)),
        "percentile_75": float(np.percentile(costs, 75)),
        "percentile_95": float(np.percentile(costs, 95)),
        "failure_probability": p_failure,
        "mean_E_rms_deg": float(np.mean(df_trials['E_rms_deg'])),
        "mean_E_peak_deg": float(np.mean(df_trials['E_peak_deg'])),
        "mean_recovery_time_s": float(np.mean(df_trials['T_recovery_s']))
    }
    
    summary_df = pd.DataFrame([stats_summary])
    summary_csv = os.path.join(output_dir, "monte_carlo_statistical_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    
    print("\n[+] Monte Carlo Statistical Evaluation Summary:")
    print(f"    Sample Mean J: {mean_J:.3f} +/- {ci_margin:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")
    print(f"    Standard Deviation: {std_J:.3f}, Variance: {var_J:.3f}")
    print(f"    Median J: {stats_summary['median_J']:.3f}, 95th Percentile: {stats_summary['percentile_95']:.3f}")
    print(f"    Failure Probability: {p_failure * 100:.1f}%")
    print(f"    Saved to {summary_csv}")
    
    return stats_summary, df_trials

if __name__ == "__main__":
    run_monte_carlo_experiments(n_trials=40)
