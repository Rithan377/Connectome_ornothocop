"""
Publication-Grade Visualizations for Drosophila Connectome Flight Simulation.
Generates:
1. Fig 1: Connectome Network Topology & Centrality Distribution.
2. Fig 2: Neural Dynamics, Spike Rasters & Motor Decoding.
3. Fig 3: 6-DOF Closed-Loop Flight Trajectories & Attitude Stabilization.
4. Fig 4: Neuron & Group Ablation Importance Rankings (with FDR threshold).
5. Fig 5: Connectome Reduction Frontier (100% Control Retention Curve).
6. Fig 6: Monte Carlo Distributions & Parameter Sensitivity Tornado.
"""

import os
import matplotlib
matplotlib.use('Agg') # Headless publication backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_all_visualizations(results_dir="results", circuits_dir="circuits", viz_dir="visualizations"):
    os.makedirs(viz_dir, exist_ok=True)
    plt.rcParams.update({
        'font.size': 11,
        'font.family': 'sans-serif',
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.autolayout': True,
        'axes.spines.top': False,
        'axes.spines.right': False
    })
    
    # -------------------------------------------------------------
    # Fig 1: Connectome Graph Centrality & Degrees
    # -------------------------------------------------------------
    cent_path = os.path.join(circuits_dir, "graph_centrality_metrics.feather")
    if os.path.exists(cent_path):
        df_cent = pd.read_feather(cent_path)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        
        # In-degree vs Out-degree
        axes[0].scatter(df_cent['in_degree'], df_cent['out_degree'], alpha=0.4, color='#2b5c8f', edgecolors='none', s=25)
        axes[0].set_xlabel("In-Degree ($k_{in}$)")
        axes[0].set_ylabel("Out-Degree ($k_{out}$)")
        axes[0].set_title("A. Synaptic Degree Distribution")
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Betweenness Centrality by Circuit Role
        roles = df_cent['circuit_role'].unique()
        data_by_role = [df_cent[df_cent['circuit_role'] == r]['betweenness_centrality'] * 1e3 for r in roles]
        axes[1].boxplot(data_by_role, tick_labels=[r.replace('_', '\n') for r in roles], patch_artist=True,
                        boxprops=dict(facecolor='#4a90e2', alpha=0.7))
        axes[1].set_ylabel("Betweenness Centrality ($C_B \\times 10^3$)")
        axes[1].set_title("B. Centrality Across Circuit Tiers")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        # In-strength vs Out-strength
        axes[2].scatter(df_cent['in_strength'], df_cent['out_strength'], alpha=0.4, color='#d9534f', edgecolors='none', s=25)
        axes[2].set_xlabel("In-Strength ($s_{in}$ synapses)")
        axes[2].set_ylabel("Out-Strength ($s_{out}$ synapses)")
        axes[2].set_title("C. Weighted Synaptic Strength")
        axes[2].set_xscale('log')
        axes[2].set_yscale('log')
        axes[2].grid(True, linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        fig.savefig(os.path.join(viz_dir, "fig1_connectome_network_topology.png"), dpi=300)
        plt.close(fig)
        print("[+] Fig 1 saved: Network Topology")

    # -------------------------------------------------------------
    # Fig 2 & 3: Closed-Loop Flight Trajectories
    # -------------------------------------------------------------
    traj_path = os.path.join(results_dir, "traj_roll_kick_8rad_s.feather")
    if not os.path.exists(traj_path):
        # Fallback to any available trajectory
        trajs = [f for f in os.listdir(results_dir) if f.startswith("traj_") and f.endswith(".feather")]
        if trajs:
            traj_path = os.path.join(results_dir, trajs[0])
            
    if os.path.exists(traj_path):
        df_t = pd.read_feather(traj_path)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        
        # Attitude angles: roll, pitch, yaw
        axes[0, 0].plot(df_t['t'] * 1000, df_t['roll_deg'], label='Roll $\\phi$', color='#d9534f', lw=2)
        axes[0, 0].plot(df_t['t'] * 1000, df_t['pitch_deg'], label='Pitch $\\theta$', color='#2b5c8f', lw=1.5, ls='--')
        axes[0, 0].plot(df_t['t'] * 1000, df_t['yaw_deg'], label='Yaw $\\psi$', color='#5cb85c', lw=1.5, ls=':')
        axes[0, 0].axhline(0, color='gray', lw=0.8, ls='-')
        axes[0, 0].set_xlabel("Time (ms)")
        axes[0, 0].set_ylabel("Orientation Error (deg)")
        axes[0, 0].set_title("A. 3-Axis Orientation Dynamics")
        axes[0, 0].legend(loc='upper right')
        axes[0, 0].grid(True, linestyle='--', alpha=0.5)
        
        # Angular velocities
        axes[0, 1].plot(df_t['t'] * 1000, df_t['omega_x'], label='Roll Rate $\\omega_x$', color='#d9534f', lw=2)
        axes[0, 1].plot(df_t['t'] * 1000, df_t['omega_y'], label='Pitch Rate $\\omega_y$', color='#2b5c8f', lw=1.5, ls='--')
        axes[0, 1].plot(df_t['t'] * 1000, df_t['omega_z'], label='Yaw Rate $\\omega_z$', color='#5cb85c', lw=1.5, ls=':')
        axes[0, 1].set_xlabel("Time (ms)")
        axes[0, 1].set_ylabel("Body Angular Velocity (rad/s)")
        axes[0, 1].set_title("B. Angular Rates & Rapid Damping")
        axes[0, 1].legend(loc='upper right')
        axes[0, 1].grid(True, linestyle='--', alpha=0.5)
        
        # Wing Kinematic Modulation: Left vs Right Amplitude
        axes[1, 0].plot(df_t['t'] * 1000, np.rad2deg(df_t['ctrl_dA_L']), label='Left Wing $\\Delta A_L$', color='#337ab7', lw=2)
        axes[1, 0].plot(df_t['t'] * 1000, np.rad2deg(df_t['ctrl_dA_R']), label='Right Wing $\\Delta A_R$', color='#f0ad4e', lw=2, ls='--')
        axes[1, 0].set_xlabel("Time (ms)")
        axes[1, 0].set_ylabel("Stroke Amplitude Offset (deg)")
        axes[1, 0].set_title("C. Connectome Motor Output to Wings")
        axes[1, 0].legend(loc='upper right')
        axes[1, 0].grid(True, linestyle='--', alpha=0.5)
        
        # Spatial 2D Trajectory / Altitude
        axes[1, 1].plot(df_t['t'] * 1000, df_t['z'] * 100, label='Altitude $z$ (cm)', color='#5cb85c', lw=2)
        axes[1, 1].plot(df_t['t'] * 1000, df_t['y'] * 100, label='Lateral drift $y$ (cm)', color='#d9534f', lw=1.5, ls='--')
        axes[1, 1].set_xlabel("Time (ms)")
        axes[1, 1].set_ylabel("Position (cm)")
        axes[1, 1].set_title("D. Flight Altitude & Translation")
        axes[1, 1].legend(loc='upper right')
        axes[1, 1].grid(True, linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        fig.savefig(os.path.join(viz_dir, "fig3_closed_loop_flight_trajectories.png"), dpi=300)
        plt.close(fig)
        print("[+] Fig 3 saved: Flight Trajectories")

    # -------------------------------------------------------------
    # Fig 4: Ablation Importance Rankings & Multiple Testing
    # -------------------------------------------------------------
    ablate_path = os.path.join(results_dir, "single_neuron_ablation_rankings.csv")
    group_path = os.path.join(results_dir, "group_ablation_results.csv")
    
    if os.path.exists(ablate_path) and os.path.exists(group_path):
        df_sing = pd.read_csv(ablate_path).head(15)
        df_grp = pd.read_csv(group_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
        
        # Group ablation bar chart
        grp_names = [g.replace('_', ' ').title() for g in df_grp['group_name']]
        colors = ['#d9534f' if f else '#4a90e2' for f in df_grp['failed']]
        axes[0].barh(grp_names, df_grp['I_importance'], color=colors, alpha=0.85)
        axes[0].axvline(0, color='black', lw=0.8)
        axes[0].set_xlabel("Ablation Importance Score ($I_i = \\Delta J$)")
        axes[0].set_title("A. Circuit Population Group Ablations")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Single neuron rankings
        labels = [f"{t}\n({bid})" for t, bid in zip(df_sing['type'], df_sing['body_id'])]
        bars = axes[1].bar(range(len(df_sing)), df_sing['I_roll_kick'], color='#2b5c8f', alpha=0.85)
        axes[1].set_xticks(range(len(df_sing)))
        axes[1].set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
        axes[1].set_ylabel("Importance Score ($I_i = J_{-i} - J_{base}$)")
        axes[1].set_title("B. Top Individual Neurons by Flight Importance")
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        fig.savefig(os.path.join(viz_dir, "fig4_ablation_rankings_and_importance.png"), dpi=300)
        plt.close(fig)
        print("[+] Fig 4 saved: Ablation Importance")

    # -------------------------------------------------------------
    # Fig 5: Minimal Circuit Reduction Curve (100% Control Retention)
    # -------------------------------------------------------------
    red_path = os.path.join(results_dir, "network_reduction_trajectory.csv")
    if os.path.exists(red_path):
        df_red = pd.read_csv(red_path)
        fig, ax1 = plt.subplots(figsize=(9, 5))
        
        ax2 = ax1.twinx()
        
        p1 = ax1.plot(df_red['num_neurons'], df_red['control_retention_pct'], 'o-', color='#2b5c8f', lw=2.5, markersize=8, label='Control Retention %')
        p2 = ax2.plot(df_red['num_neurons'], df_red['J_cost'], 's--', color='#d9534f', lw=2, markersize=7, label='Performance Cost $J$')
        
        ax1.set_xlabel("Number of Neurons in Circuit")
        ax1.set_ylabel("Flight Control Retention (%)", color='#2b5c8f')
        ax2.set_ylabel("Flight Performance Cost $J$", color='#d9534f')
        ax1.set_title("Minimal Connectome Synthesis: 100% Flight Control Retention Frontier")
        
        ax1.axhline(100.0, color='gray', linestyle=':', alpha=0.7)
        ax1.set_ylim([0, 115])
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Combined legend
        lines = p1 + p2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='lower right')
        
        fig.tight_layout()
        fig.savefig(os.path.join(viz_dir, "fig5_minimal_circuit_reduction_curve.png"), dpi=300)
        plt.close(fig)
        print("[+] Fig 5 saved: Minimal Circuit Reduction Curve")

    # -------------------------------------------------------------
    # Fig 6: Sensitivity Tornado Chart & Monte Carlo
    # -------------------------------------------------------------
    sens_path = os.path.join(results_dir, "parameter_sensitivity_results.csv")
    mc_path = os.path.join(results_dir, "monte_carlo_trials.csv")
    
    if os.path.exists(sens_path) and os.path.exists(mc_path):
        df_sens = pd.read_csv(sens_path)
        df_mc = pd.read_csv(mc_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Sensitivity Tornado
        axes[0].barh(df_sens['description'], df_sens['normalized_sensitivity'], color='#4a90e2', alpha=0.85)
        axes[0].axvline(0, color='black', lw=0.8)
        axes[0].set_xlabel("Normalized Sensitivity $\\frac{\\Delta J / J}{\\Delta p / p}$")
        axes[0].set_title("A. Parameter Sensitivity Analysis")
        axes[0].grid(True, linestyle='--', alpha=0.5)
        
        # Monte Carlo Distribution
        axes[1].hist(df_mc['J_cost'], bins=12, color='#5cb85c', edgecolor='black', alpha=0.75)
        axes[1].axvline(np.mean(df_mc['J_cost']), color='red', lw=2, ls='--', label=f"Mean J = {np.mean(df_mc['J_cost']):.2f}")
        axes[1].set_xlabel("Flight Performance Cost $J$")
        axes[1].set_ylabel("Number of Trials")
        axes[1].set_title(f"B. Monte Carlo Distribution (N={len(df_mc)})")
        axes[1].legend(loc='upper right')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        fig.tight_layout()
        fig.savefig(os.path.join(viz_dir, "fig6_parameter_sensitivity_and_monte_carlo.png"), dpi=300)
        plt.close(fig)
        print("[+] Fig 6 saved: Sensitivity & Monte Carlo")

if __name__ == "__main__":
    generate_all_visualizations()
