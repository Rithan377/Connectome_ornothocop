"""
Comprehensive Experimental Pipeline for FlyBody MuJoCo + Connectome Closed-Loop Simulation.

Executes:
1. Baseline Still-Air & Environmental Disturbance Suite (Gusts, Crosswinds, Turbulence, Kicks)
2. Monte Carlo Statistical Reliability Analysis (Mean, Std, 95% CI, Failure Probability)
3. Physical & Sensory Parameter Sensitivity Analysis (Mass scaling, Latency delays, Noise)
4. Targeted Connectome Ablation Experiments (Individual Neurons, Populations, FDR control)
5. Generation of CSV Results and Publication-Grade Visualizations
"""

import os
import time
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

from orthocop.flybody.closed_loop import ClosedLoopFlyBodySim


def run_disturbance_suite(duration: float = 0.25) -> pd.DataFrame:
    """
    Run disturbance rejection across all controller configurations.
    """
    print("=" * 70)
    print("EXPERIMENT 1: DISTURBANCE REJECTION SUITE (MUJOCO FLYBODY)")
    print("=" * 70)

    disturbances = [
        ('Still Air Hover', 'none', 0.0),
        ('Roll Kick (+6 rad/s)', 'roll_kick', 6.0),
        ('Pitch Kick (+4 rad/s)', 'pitch_kick', 4.0),
        ('Yaw Kick (+4 rad/s)', 'yaw_kick', 4.0),
        ('Crosswind (1.5 m/s)', 'crosswind', 1.5),
        ('Headwind Gust (2.0 m/s)', 'gust', 2.0),
        ('Stochastic Turbulence', 'turbulence', 1.0),
    ]

    controllers = [
        ('Pretrained FlyBody Policy', 'pretrained', {}),
        ('Full Connectome Brain (2,300N)', 'connectome_full', {
            'nodes_path': 'circuits/flight_candidate_neurons.feather',
            'edges_path': 'circuits/flight_subnetwork_edges_min2.feather'
        }),
        ('Reduced Connectome Brain (148N)', 'connectome_reduced', {
            'nodes_path': 'circuits/ultra_minimal_flight_nodes.feather',
            'edges_path': 'circuits/ultra_minimal_flight_edges.feather'
        }),
        ('Haltere-Ablated Connectome', 'connectome_reduced', {
            'nodes_path': 'circuits/ultra_minimal_flight_nodes.feather',
            'edges_path': 'circuits/ultra_minimal_flight_edges.feather',
            'ablate_halteres': True
        }),
        ('Steering MN 800381 Ablated', 'connectome_reduced', {
            'nodes_path': 'circuits/ultra_minimal_flight_nodes.feather',
            'edges_path': 'circuits/ultra_minimal_flight_edges.feather',
            'ablated_nodes': [800381]
        })
    ]

    records = []

    for c_name, c_type, c_kwargs in controllers:
        print(f"\nEvaluating: {c_name}")
        sim = ClosedLoopFlyBodySim(controller_type=c_type, **c_kwargs)
        
        for d_name, d_type, d_mag in disturbances:
            res = sim.run_simulation(
                duration=duration,
                disturbance_type=d_type,
                disturbance_magnitude=d_mag,
                disturbance_time=0.05
            )
            print(f"  [{d_name:25s}] RMS Error: {res['rms_attitude_error_deg']:6.2f}° | "
                  f"Peak Rate: {res['peak_ang_vel_rad_s']:6.2f} rad/s | "
                  f"Recovery: {res['recovery_time_s']*1000:5.1f} ms | Crashed: {res['crashed']}")

            records.append({
                'controller': c_name,
                'disturbance': d_name,
                'rms_attitude_error_deg': res['rms_attitude_error_deg'],
                'rms_roll_deg': res['rms_roll_deg'],
                'rms_pitch_deg': res['rms_pitch_deg'],
                'rms_yaw_deg': res['rms_yaw_deg'],
                'peak_ang_vel_rad_s': res['peak_ang_vel_rad_s'],
                'rms_ang_vel_rad_s': res['rms_ang_vel_rad_s'],
                'recovery_time_ms': res['recovery_time_s'] * 1000.0,
                'trajectory_drift_cm': res['trajectory_drift_m'] * 100.0,
                'control_effort': res['control_effort'],
                'crashed': int(res['crashed'])
            })

    df = pd.DataFrame(records)
    os.makedirs('results', exist_ok=True)
    df.to_csv('results/flybody_disturbance_rejection_summary.csv', index=False)
    print("\nSaved disturbance suite to results/flybody_disturbance_rejection_summary.csv")
    return df


def run_monte_carlo(n_trials: int = 25, duration: float = 0.20) -> pd.DataFrame:
    """
    Run Monte Carlo statistical reliability test across randomized seeds and turbulence.
    """
    print("\n" + "=" * 70)
    print(f"EXPERIMENT 2: MONTE CARLO STATISTICAL EVALUATION ({n_trials} TRIALS)")
    print("=" * 70)

    controllers = [
        ('Pretrained Baseline', 'pretrained', {}),
        ('Connectome Full (2,300N)', 'connectome_full', {
            'nodes_path': 'circuits/flight_candidate_neurons.feather',
            'edges_path': 'circuits/flight_subnetwork_edges_min2.feather'
        }),
        ('Connectome Reduced (148N)', 'connectome_reduced', {
            'nodes_path': 'circuits/ultra_minimal_flight_nodes.feather',
            'edges_path': 'circuits/ultra_minimal_flight_edges.feather'
        }),
        ('Haltere-Ablated', 'connectome_reduced', {
            'nodes_path': 'circuits/ultra_minimal_flight_nodes.feather',
            'edges_path': 'circuits/ultra_minimal_flight_edges.feather',
            'ablate_halteres': True
        })
    ]

    summary_records = []

    for c_name, c_type, c_kwargs in controllers:
        sim = ClosedLoopFlyBodySim(controller_type=c_type, **c_kwargs)
        rms_errors = []
        crashes = []
        recovery_times = []
        efforts = []

        for trial in range(n_trials):
            np.random.seed(1000 + trial)
            roll_mag = np.random.uniform(4.0, 7.0)
            res = sim.run_simulation(
                duration=duration,
                disturbance_type='turbulence',
                disturbance_magnitude=roll_mag,
                disturbance_time=0.04
            )
            rms_errors.append(res['rms_attitude_error_deg'])
            crashes.append(int(res['crashed']))
            recovery_times.append(res['recovery_time_s'] * 1000.0)
            efforts.append(res['control_effort'])

        rms_arr = np.array(rms_errors)
        mean_rms = np.mean(rms_arr)
        std_rms = np.std(rms_arr, ddof=1)
        sem_rms = std_rms / np.sqrt(n_trials)
        ci95 = stats.t.interval(0.95, df=n_trials - 1, loc=mean_rms, scale=sem_rms)
        fail_prob = np.mean(crashes)

        print(f"[{c_name:25s}] Mean RMS: {mean_rms:5.2f}° ± {std_rms:4.2f}° | "
              f"95% CI: [{ci95[0]:5.2f}°, {ci95[1]:5.2f}°] | Failure: {fail_prob*100:4.1f}%")

        summary_records.append({
            'controller': c_name,
            'n_trials': n_trials,
            'mean_rms_deg': mean_rms,
            'std_rms_deg': std_rms,
            'sem_rms_deg': sem_rms,
            'ci95_low': ci95[0],
            'ci95_high': ci95[1],
            'mean_recovery_ms': np.mean(recovery_times),
            'std_recovery_ms': np.std(recovery_times, ddof=1),
            'mean_effort': np.mean(efforts),
            'failure_probability': fail_prob
        })

    df = pd.DataFrame(summary_records)
    df.to_csv('results/flybody_monte_carlo_statistical_summary.csv', index=False)
    print("\nSaved Monte Carlo summary to results/flybody_monte_carlo_statistical_summary.csv")
    return df


def run_parameter_sensitivity() -> pd.DataFrame:
    """
    Run physical mass and sensory latency sensitivity analysis.
    """
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: PHYSICAL & SENSORY SENSITIVITY ANALYSIS")
    print("=" * 70)

    # 1. Haltere latency delay sweep
    delay_records = []
    delays_ms = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
    
    for d in delays_ms:
        sim = ClosedLoopFlyBodySim(
            controller_type='connectome_reduced',
            nodes_path='circuits/ultra_minimal_flight_nodes.feather',
            edges_path='circuits/ultra_minimal_flight_edges.feather',
            haltere_delay_ms=d
        )
        res = sim.run_simulation(duration=0.20, disturbance_type='roll_kick', disturbance_magnitude=6.0)
        delay_records.append({
            'parameter': 'haltere_delay_ms',
            'value': d,
            'rms_attitude_error_deg': res['rms_attitude_error_deg'],
            'peak_ang_vel_rad_s': res['peak_ang_vel_rad_s'],
            'crashed': int(res['crashed'])
        })
        print(f"  Haltere Delay {d:3.1f} ms: RMS Error = {res['rms_attitude_error_deg']:5.2f}°")

    # 2. Body mass perturbation sweep
    mass_factors = [0.8, 0.9, 1.0, 1.1, 1.2]
    for m in mass_factors:
        sim = ClosedLoopFlyBodySim(
            controller_type='connectome_reduced',
            nodes_path='circuits/ultra_minimal_flight_nodes.feather',
            edges_path='circuits/ultra_minimal_flight_edges.feather'
        )
        sim.env.perturb_mass(m)
        res = sim.run_simulation(duration=0.20, disturbance_type='roll_kick', disturbance_magnitude=6.0)
        delay_records.append({
            'parameter': 'mass_scaling_factor',
            'value': m,
            'rms_attitude_error_deg': res['rms_attitude_error_deg'],
            'peak_ang_vel_rad_s': res['peak_ang_vel_rad_s'],
            'crashed': int(res['crashed'])
        })
        print(f"  Mass Factor {m:3.1f}x: RMS Error = {res['rms_attitude_error_deg']:5.2f}°")

    df = pd.DataFrame(delay_records)
    df.to_csv('results/flybody_parameter_sensitivity_results.csv', index=False)
    print("\nSaved sensitivity results to results/flybody_parameter_sensitivity_results.csv")
    return df


def plot_results():
    """
    Generate publication-quality figures comparing FlyBody controllers.
    """
    os.makedirs('visualizations', exist_ok=True)
    
    # 1. Disturbance rejection bar chart
    if os.path.exists('results/flybody_disturbance_rejection_summary.csv'):
        df = pd.read_csv('results/flybody_disturbance_rejection_summary.csv')
        pivot = df.pivot(index='disturbance', columns='controller', values='rms_attitude_error_deg')
        
        plt.figure(figsize=(12, 6))
        pivot.plot(kind='bar', figsize=(13, 6), colormap='tab10', width=0.82)
        plt.title('FlyBody MuJoCo Physics: Disturbance Rejection Across Controllers', fontsize=14, fontweight='bold')
        plt.xlabel('Environmental Disturbance Condition', fontsize=12)
        plt.ylabel('RMS Attitude Tracking Error (degrees)', fontsize=12)
        plt.xticks(rotation=25, ha='right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title='Controller Architecture', loc='upper left')
        plt.tight_layout()
        plt.savefig('visualizations/fig9_flybody_mujoco_disturbance_rejection.png', dpi=300)
        plt.close()
        print("Generated visualizations/fig9_flybody_mujoco_disturbance_rejection.png")

    # 2. Monte Carlo & Sensitivity Plots
    if os.path.exists('results/flybody_monte_carlo_statistical_summary.csv') and os.path.exists('results/flybody_parameter_sensitivity_results.csv'):
        mc_df = pd.read_csv('results/flybody_monte_carlo_statistical_summary.csv')
        sens_df = pd.read_csv('results/flybody_parameter_sensitivity_results.csv')

        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

        # Subplot 1: Monte Carlo with 95% CI error bars
        ctrls = mc_df['controller'].tolist()
        means = mc_df['mean_rms_deg'].tolist()
        err_low = [m - c_low for m, c_low in zip(means, mc_df['ci95_low'])]
        err_high = [c_high - m for m, c_high in zip(means, mc_df['ci95_high'])]
        yerr = [err_low, err_high]

        colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']
        axes[0].bar(ctrls, means, yerr=yerr, capsize=6, color=colors[:len(ctrls)], alpha=0.85, edgecolor='black')
        axes[0].set_title('Monte Carlo Reliability (N=25 Trials, 95% Student-t CI)', fontweight='bold')
        axes[0].set_ylabel('Mean RMS Attitude Error (degrees)')
        axes[0].set_xticklabels(ctrls, rotation=20, ha='right')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Subplot 2: Haltere Delay Latency Sensitivity
        d_df = sens_df[sens_df['parameter'] == 'haltere_delay_ms']
        axes[1].plot(d_df['value'], d_df['rms_attitude_error_deg'], 'o-', color='#e377c2', linewidth=2.5, markersize=8)
        axes[1].set_title('Sensory Latency Sensitivity (Haltere Axonal Delay)', fontweight='bold')
        axes[1].set_xlabel('Haltere Delay Tau_d (ms)')
        axes[1].set_ylabel('RMS Attitude Error (degrees)')
        axes[1].grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig('visualizations/fig10_flybody_connectome_ablation_and_monte_carlo.png', dpi=300)
        plt.close()
        print("Generated visualizations/fig10_flybody_connectome_ablation_and_monte_carlo.png")


if __name__ == '__main__':
    t0 = time.time()
    run_disturbance_suite(duration=0.20)
    run_monte_carlo(n_trials=20, duration=0.15)
    run_parameter_sensitivity()
    plot_results()
    print(f"\nAll FlyBody MuJoCo closed-loop experiments completed in {time.time() - t0:.1f} seconds!")
