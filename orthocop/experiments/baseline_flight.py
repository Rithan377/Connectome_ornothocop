"""
Baseline Flight and Disturbance Rejection Experiments.
Evaluates the connectome flight controller in still air and under impulsive kicks and wind gusts.
"""

import os
import json
import numpy as np
import pandas as pd
from orthocop.simulations.closed_loop import ClosedLoopFlightSimulator

def run_baseline_and_disturbances(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    nodes = pd.read_feather("circuits/flight_candidate_neurons.feather")
    edges = pd.read_feather("circuits/flight_subnetwork_edges_min2.feather")
    
    sim = ClosedLoopFlightSimulator(nodes_df=nodes, edges_df=edges)
    
    experiments = [
        {"name": "still_air_baseline", "type": "none", "mag": 0.0, "time": 0.02, "wind": np.zeros(3)},
        {"name": "roll_kick_8rad_s", "type": "roll_kick", "mag": 8.0, "time": 0.02, "wind": np.zeros(3)},
        {"name": "pitch_kick_6rad_s", "type": "pitch_kick", "mag": 6.0, "time": 0.02, "wind": np.zeros(3)},
        {"name": "yaw_kick_6rad_s", "type": "yaw_kick", "mag": 6.0, "time": 0.02, "wind": np.zeros(3)},
        {"name": "gust_crosswind_2m_s", "type": "gust", "mag": 2.0, "time": 0.02, "wind": np.zeros(3)},
        {"name": "steady_crosswind_1m_s", "type": "none", "mag": 0.0, "time": 0.02, "wind": np.array([0.0, 1.0, 0.0])},
        {"name": "steady_headwind_1m_s", "type": "none", "mag": 0.0, "time": 0.02, "wind": np.array([1.0, 0.0, 0.0])}
    ]
    
    summary_results = []
    
    for exp in experiments:
        print(f"[*] Running experiment: {exp['name']}...")
        df_traj, metrics = sim.run_simulation(
            duration_s=0.10,
            disturbance_type=exp['type'],
            disturbance_mag=exp['mag'],
            disturbance_time=exp['time'],
            u_wind_base=exp['wind']
        )
        traj_file = os.path.join(output_dir, f"traj_{exp['name']}.feather")
        df_traj.to_feather(traj_file)
        
        metrics['experiment'] = exp['name']
        summary_results.append(metrics)
        print(f"    E_rms: {metrics['E_rms_deg']:.2f} deg, E_peak: {metrics['E_peak_deg']:.2f} deg, J: {metrics['J_cost']:.2f}, Failed: {metrics['failed']}")
        
    df_summary = pd.DataFrame(summary_results)
    out_csv = os.path.join(output_dir, "disturbance_rejection_summary.csv")
    df_summary.to_csv(out_csv, index=False)
    print(f"\n[+] All baseline and disturbance experiments finished. Saved to {out_csv}")
    return df_summary

if __name__ == "__main__":
    run_baseline_and_disturbances()
