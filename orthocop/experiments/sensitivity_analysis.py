"""
Finite-Difference Parameter Sensitivity Analysis.
Quantifies the influence of neural, physical, and sensory parameters on closed-loop flight performance cost J:
S_p = (J(p + Delta p) - J(p)) / Delta p
Normalized Sensitivity = (Delta J / J_base) / (Delta p / p_base).
"""

import os
import numpy as np
import pandas as pd
from orthocop.simulations.closed_loop import ClosedLoopFlightSimulator

def run_sensitivity_analysis(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    nodes = pd.read_feather("circuits/flight_candidate_neurons.feather")
    edges = pd.read_feather("circuits/flight_subnetwork_edges_min2.feather")
    
    # 1. Baseline simulation
    print("[*] Running Baseline simulation for Sensitivity Analysis...")
    base_sim = ClosedLoopFlightSimulator(nodes_df=nodes, edges_df=edges)
    _, base_metrics = base_sim.run_simulation(
        duration_s=0.08,
        disturbance_type='roll_kick',
        disturbance_mag=8.0,
        disturbance_time=0.015
    )
    J_base = base_metrics['J_cost']
    print(f"[+] Baseline J_cost = {J_base:.4f}")
    
    # Parameters to perturb
    param_variations = [
        {"param": "wingbeat_frequency", "base": 200.0, "delta_pct": 0.10, "desc": "Wingbeat Flapping Frequency (Hz)"},
        {"param": "body_mass", "base": 1.0e-6, "delta_pct": 0.10, "desc": "Total Body Mass (kg)"},
        {"param": "inertia_Ixx", "base": 1.5e-13, "delta_pct": 0.10, "desc": "Roll Moment of Inertia (kg*m^2)"},
        {"param": "sensor_delay", "base": 0.0015, "delta_pct": 0.20, "desc": "Haltere Sensor Delay (s)"},
        {"param": "sensor_noise", "base": 0.05, "delta_pct": 0.20, "desc": "Haltere Measurement Noise Std"},
        {"param": "aerodynamic_CL", "base": 1.8, "delta_pct": 0.10, "desc": "Max Flapping Lift Coefficient CL"}
    ]
    
    sensitivity_records = []
    
    for item in param_variations:
        p_name = item["param"]
        p_val = item["base"]
        dp_frac = item["delta_pct"]
        dp = p_val * dp_frac
        p_perturbed = p_val + dp
        
        print(f"[*] Testing sensitivity for {item['desc']} (+{dp_frac*100:.0f}%)...")
        
        # Instantiate simulator and apply parameter perturbation
        sim_p = ClosedLoopFlightSimulator(nodes_df=nodes, edges_df=edges)
        
        if p_name == "wingbeat_frequency":
            sim_p.physics.base_frequency = p_perturbed
        elif p_name == "body_mass":
            sim_p.physics.mass = p_perturbed
        elif p_name == "inertia_Ixx":
            sim_p.physics.I[0, 0] = p_perturbed
            sim_p.physics.I_inv = np.linalg.inv(sim_p.physics.I)
        elif p_name == "sensor_delay":
            sim_p.sensors.haltere_roll.delay_steps = int(round(p_perturbed / sim_p.dt))
        elif p_name == "sensor_noise":
            sim_p.sensors.haltere_roll.noise_std = p_perturbed
        elif p_name == "aerodynamic_CL":
            sim_p.physics.CL_max = p_perturbed
            
        _, p_metrics = sim_p.run_simulation(
            duration_s=0.08,
            disturbance_type='roll_kick',
            disturbance_mag=8.0,
            disturbance_time=0.015
        )
        
        J_perturbed = p_metrics['J_cost']
        dJ = J_perturbed - J_base
        raw_sensitivity = dJ / dp
        normalized_sensitivity = (dJ / J_base) / dp_frac
        
        sensitivity_records.append({
            "parameter": p_name,
            "description": item["desc"],
            "base_value": p_val,
            "perturbed_value": p_perturbed,
            "delta_p": dp,
            "J_base": J_base,
            "J_perturbed": J_perturbed,
            "delta_J": dJ,
            "raw_sensitivity": raw_sensitivity,
            "normalized_sensitivity": normalized_sensitivity
        })
        
        print(f"    dJ: {dJ:+.4f}, Normalized Sensitivity: {normalized_sensitivity:+.3f}")
        
    df_sens = pd.DataFrame(sensitivity_records).sort_values(by="normalized_sensitivity", key=abs, ascending=False)
    out_csv = os.path.join(output_dir, "parameter_sensitivity_results.csv")
    df_sens.to_csv(out_csv, index=False)
    print(f"\n[+] Parameter Sensitivity analysis complete. Saved to {out_csv}")
    return df_sens

if __name__ == "__main__":
    run_sensitivity_analysis()
