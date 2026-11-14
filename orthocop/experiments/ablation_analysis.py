"""
Neuron and Synaptic Ablation Analysis.
Implements:
1. Single neuron ablation importance score: I_i = J_{-i} - J_baseline
2. Context-dependent ablation: I_i(c) across still air, roll kick, and wind gust.
3. Group ablation (Haltere sensory, Wing mechanosensory, Visual LPTC, Descending Neurons, Premotor, Motor).
4. Synaptic perturbation scaling: w' = alpha * w.
5. Random ablation controls.
6. Multiple testing statistical correction (Benjamini-Hochberg FDR control).
"""

import os
import json
import numpy as np
import pandas as pd
from scipy import stats
from orthocop.simulations.closed_loop import ClosedLoopFlightSimulator

def run_ablation_experiments(output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)
    nodes = pd.read_feather("circuits/flight_candidate_neurons.feather")
    edges = pd.read_feather("circuits/flight_subnetwork_edges_min2.feather")
    
    # Baseline simulation
    print("[*] Running Baseline simulation for Ablation Analysis...")
    sim_base = ClosedLoopFlightSimulator(nodes_df=nodes, edges_df=edges)
    _, base_metrics = sim_base.run_simulation(
        duration_s=0.08,
        disturbance_type='roll_kick',
        disturbance_mag=8.0,
        disturbance_time=0.015
    )
    J_base = base_metrics['J_cost']
    print(f"[+] Baseline J_cost: {J_base:.4f} (E_rms: {base_metrics['E_rms_deg']:.2f} deg, T_recov: {base_metrics['T_recovery_s']:.3f} s)")
    
    # 1. GROUP ABLATION EXPERIMENTS
    print("\n[*] Running Group Ablation Experiments...")
    group_defs = {
        "haltere_sensory": nodes[nodes['circuit_role'] == 'haltere_sensory']['body_id'].tolist(),
        "wing_sensory": nodes[nodes['circuit_role'] == 'wing_sensory']['body_id'].tolist(),
        "visual_sensory": nodes[nodes['circuit_role'] == 'visual_sensory']['body_id'].tolist(),
        "descending_neurons": nodes[nodes['circuit_role'] == 'descending_neuron']['body_id'].tolist(),
        "premotor_interneurons": nodes[nodes['circuit_role'] == 'premotor_interneuron']['body_id'].tolist(),
        "steering_motor_neurons": nodes[nodes['type'].str.contains('b1|b2|i1|i2', case=False, na=False)]['body_id'].tolist(),
        "random_control_100_neurons": np.random.choice(nodes['body_id'].values, size=100, replace=False).tolist()
    }
    
    group_results = []
    for gname, g_ids in group_defs.items():
        print(f"[*] Testing Group Ablation: {gname} ({len(g_ids)} neurons ablated)...")
        sim_group = ClosedLoopFlightSimulator(nodes_df=nodes, edges_df=edges, ablated_neurons=g_ids)
        _, g_metrics = sim_group.run_simulation(
            duration_s=0.08,
            disturbance_type='roll_kick',
            disturbance_mag=8.0,
            disturbance_time=0.015
        )
        J_group = g_metrics['J_cost']
        I_group = J_group - J_base
        g_metrics['group_name'] = gname
        g_metrics['num_ablated'] = len(g_ids)
        g_metrics['I_importance'] = float(I_group)
        group_results.append(g_metrics)
        print(f"    J: {J_group:.4f}, Importance I: {I_group:+.4f}, Failed: {g_metrics['failed']}")
        
    df_group = pd.DataFrame(group_results).sort_values(by="I_importance", ascending=False)
    df_group.to_csv(os.path.join(output_dir, "group_ablation_results.csv"), index=False)
    
    # 2. SYNAPTIC PERTURBATION EXPERIMENT: w' = alpha * w
    print("\n[*] Running Synaptic Perturbation Experiments (scaling alpha)...")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    syn_results = []
    for alpha in alphas:
        mod_edges = edges.copy()
        mod_edges['weight'] = mod_edges['weight'] * alpha
        sim_syn = ClosedLoopFlightSimulator(nodes_df=nodes, edges_df=mod_edges)
        _, s_metrics = sim_syn.run_simulation(
            duration_s=0.08,
            disturbance_type='roll_kick',
            disturbance_mag=8.0,
            disturbance_time=0.015
        )
        s_metrics['alpha'] = alpha
        s_metrics['delta_J'] = float(s_metrics['J_cost'] - J_base)
        syn_results.append(s_metrics)
        print(f"    alpha={alpha:.2f}: J={s_metrics['J_cost']:.4f}, delta_J={s_metrics['delta_J']:+.4f}, Failed={s_metrics['failed']}")
    df_syn = pd.DataFrame(syn_results)
    df_syn.to_csv(os.path.join(output_dir, "synaptic_perturbation_results.csv"), index=False)
    
    # 3. SINGLE NEURON ABLATION ON KEY CANDIDATES & STATISTICAL MULTIPLE TESTING
    print("\n[*] Running Single Neuron Ablation on top candidates across circuit tiers...")
    # Select sample of high-centrality and diverse functional neurons
    top_candidates = [
        # Major Descending Neurons
        10001, 10010, 11466, 10247, 10544,
        # Steering Motor Neurons
        800381, 800573, 800637, 800899,
        # Haltere Afferents
        29495, 36718, 41670, 52279,
        # Premotor Interneurons
        800103, 800211, 801296,
        # Random controls
    ]
    # Filter to existing in nodes
    avail_candidates = [c for c in top_candidates if c in set(nodes['body_id'])]
    # Add 10 random control neurons
    random_controls = np.random.choice([b for b in nodes['body_id'] if b not in avail_candidates], size=10, replace=False)
    eval_candidates = list(avail_candidates) + list(random_controls)
    
    node_meta = nodes.set_index('body_id').to_dict('index')
    single_ablation_results = []
    
    # Contexts: still air, roll kick, gust
    contexts = [
        {"name": "roll_kick", "type": "roll_kick", "mag": 8.0},
        {"name": "still_air", "type": "none", "mag": 0.0},
        {"name": "wind_gust", "type": "gust", "mag": 2.0}
    ]
    
    # Get baseline cost for each context
    context_baselines = {}
    for ctx in contexts:
        _, c_met = sim_base.run_simulation(duration_s=0.08, disturbance_type=ctx['type'], disturbance_mag=ctx['mag'], disturbance_time=0.015)
        context_baselines[ctx['name']] = c_met['J_cost']
        
    for idx, bid in enumerate(eval_candidates):
        is_control = bid in set(random_controls)
        meta = node_meta.get(bid, {})
        c_role = meta.get('circuit_role', 'random_control')
        c_type = meta.get('type', 'unknown')
        
        row_res = {
            'body_id': int(bid),
            'type': c_type,
            'circuit_role': c_role,
            'is_control': is_control
        }
        
        # Test in each context
        sim_ablate = ClosedLoopFlightSimulator(nodes_df=nodes, edges_df=edges, ablated_neurons=[bid])
        
        for ctx in contexts:
            _, met = sim_ablate.run_simulation(
                duration_s=0.08,
                disturbance_type=ctx['type'],
                disturbance_mag=ctx['mag'],
                disturbance_time=0.015
            )
            i_score = met['J_cost'] - context_baselines[ctx['name']]
            row_res[f"I_{ctx['name']}"] = float(i_score)
            row_res[f"J_{ctx['name']}"] = float(met['J_cost'])
            row_res[f"failed_{ctx['name']}"] = met['failed']
            
        single_ablation_results.append(row_res)
        if (idx + 1) % 5 == 0 or idx == len(eval_candidates) - 1:
            print(f"    Completed {idx + 1}/{len(eval_candidates)} single neuron ablations...")
            
    df_single = pd.DataFrame(single_ablation_results)
    
    # Multiple testing correction using Benjamini-Hochberg (FDR) on roll kick importance
    # Null hypothesis: neuron ablation importance is zero (sampled from random control distribution)
    ctrl_scores = df_single[df_single['is_control']]['I_roll_kick'].values
    ctrl_mean = np.mean(ctrl_scores) if len(ctrl_scores) > 0 else 0.0
    ctrl_std = np.std(ctrl_scores, ddof=1) if len(ctrl_scores) > 1 else 0.05
    ctrl_std = max(1e-4, ctrl_std)
    
    # One-sample z-test / t-test p-value against random controls
    z_scores = (df_single['I_roll_kick'] - ctrl_mean) / ctrl_std
    p_values = 1.0 - stats.norm.cdf(z_scores)
    df_single['p_value_raw'] = p_values
    
    # Benjamini-Hochberg procedure
    m = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]
    q_values = np.zeros(m)
    min_q = 1.0
    for i in range(m - 1, -1, -1):
        q = sorted_p[i] * m / (i + 1)
        min_q = min(min_q, q)
        q_values[i] = min_q
        
    df_single['fdr_q_value'] = np.clip(q_values[np.argsort(sorted_indices)], 0.0, 1.0)
    df_single['significant_fdr_05'] = df_single['fdr_q_value'] < 0.05
    
    df_single = df_single.sort_values(by="I_roll_kick", ascending=False)
    out_single_csv = os.path.join(output_dir, "single_neuron_ablation_rankings.csv")
    df_single.to_csv(out_single_csv, index=False)
    
    print(f"\n[+] Single neuron ablation and FDR rankings complete. Saved to {out_single_csv}")
    print("Top 5 Critical Flight Neurons by Importance Score I_i:")
    print(df_single[['body_id', 'type', 'circuit_role', 'I_roll_kick', 'fdr_q_value', 'significant_fdr_05']].head(5))
    
    return df_group, df_syn, df_single

if __name__ == "__main__":
    run_ablation_experiments()
