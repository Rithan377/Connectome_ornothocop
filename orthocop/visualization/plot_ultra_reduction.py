"""
Visualizations for Ultra-Reduced Connectome Synthesis and Neuromorphic Hardware Frontier.
Generates:
Fig 7: Multi-Panel Ultra-Reduced Retention Frontier, Power Trade-offs, and Neuromorphic Specification.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_ultra_reduced_frontier(results_dir="results", viz_dir="visualizations"):
    os.makedirs(viz_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "ultra_reduced_connectome_frontier.csv")
    if not os.path.exists(csv_path):
        print(f"[!] File not found: {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # 1. Panel A: Neurons vs Retention %
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    
    p1 = ax1.plot(df['num_neurons'], df['control_retention_pct'], 'o-', color='#2b5c8f', lw=2.5, markersize=8, label='Control Retention %')
    p2 = ax1_twin.plot(df['num_neurons'], df['J_mean'], 's--', color='#d9534f', lw=2, markersize=7, label='Multi-Axis Cost $J$')
    
    ax1.axhline(100.0, color='green', ls=':', lw=1.5, alpha=0.7, label='100% Retention Threshold')
    ax1.set_xlabel("Number of Neurons in Circuit")
    ax1.set_ylabel("Flight Control Retention (%)", color='#2b5c8f')
    ax1_twin.set_ylabel("Multi-Axis Performance Cost $J$", color='#d9534f')
    ax1.set_title("A. 100% Flight Control Retention Frontier")
    ax1.set_ylim([0, 115])
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    lines = p1 + p2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='lower right')
    
    # 2. Panel B: Synaptic Compression vs Error
    ax2 = axes[1]
    ax2.plot(df['compression_pct'], df['E_rms_mean'], 'd-', color='#f0ad4e', lw=2.5, markersize=8, label='Mean $E_{RMS}$')
    ax2.plot(df['compression_pct'], df['E_peak_max'], '^--', color='#d9534f', lw=2, markersize=7, label='Max Peak Error')
    ax2.set_xlabel("Connectome Size Reduction (%)")
    ax2.set_ylabel("Attitude Error (degrees)")
    ax2.set_title("B. Dynamic Orientation Error vs Compression")
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(loc='upper left')
    
    # 3. Panel C: Neuromorphic Power & Synaptic Operations Scaling
    ax3 = axes[2]
    # Power model: 50 Hz average rate * 5 pJ per synaptic event * num_synapses
    power_uW = df['num_synapses'] * 50.0 * 5e-12 * 1e6
    sops_k = (df['num_synapses'] * 50.0) / 1000.0
    
    ax3.plot(df['num_neurons'], power_uW, 'o-', color='#5cb85c', lw=2.5, markersize=8, label='Active Neuromorphic Power ($\\mu$W)')
    ax3.set_xlabel("Number of Neurons")
    ax3.set_ylabel("Estimated Power Consumption ($\\mu$W)", color='#5cb85c')
    ax3.set_title("C. Ultra-Low Neuromorphic Power Scaling")
    ax3.grid(True, linestyle='--', alpha=0.5)
    
    # Annotate minimum viable
    min_viable = df[df['status'] == 'RETAINED_100%'].iloc[-1]
    min_power = min_viable['num_synapses'] * 50.0 * 5e-12 * 1e6
    ax3.annotate(
        f"Minimal 100% Tier:\n{int(min_viable['num_neurons'])} Neurons\n{min_power:.2f} $\\mu$W",
        xy=(min_viable['num_neurons'], min_power),
        xytext=(min_viable['num_neurons'] + 200, min_power + 3.0),
        arrowprops=dict(facecolor='black', shrink=0.08, width=1, headwidth=6),
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="#e8f4f8", ec="#2b5c8f")
    )
    
    fig.tight_layout()
    out_png = os.path.join(viz_dir, "fig7_ultra_reduced_frontier.png")
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"[+] Fig 7 saved: {out_png}")

if __name__ == "__main__":
    plot_ultra_reduced_frontier()
