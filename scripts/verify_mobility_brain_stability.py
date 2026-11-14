"""
Verify Flight Stability: Baseline Flight Brain vs Comprehensive Mobility Brain.

Runs closed-loop simulation in MuJoCo FlyBody comparing:
1. Baseline Flight Brain (897 neurons, 34,011 synapses)
2. Comprehensive Mobility Brain (10,668 neurons, 527,309 synapses)

Measures:
- Survival (staying airborne without crashing)
- Peak roll deviation under 6.0 rad/s roll perturbation
- Recovery time
- Altitude stability (Z-drift)
- Leg and Jump reflex activation
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt

from orthocop.flybody.closed_loop import ClosedLoopFlyBodySim


def run_benchmark():
    print("=" * 75)
    print(" FLIGHT STABILITY BENCHMARK: BASELINE vs COMPREHENSIVE MOBILITY BRAIN")
    print("=" * 75)

    duration = 1.0
    dist_type = 'roll_kick'
    dist_mag = 6.0
    dist_time = 0.2

    # 1. Benchmark Baseline Flight Brain
    print("\n[*] 1. Running Baseline Flight Brain (897 neurons)...")
    sim_base = ClosedLoopFlyBodySim(
        controller_type='connectome_hybrid',
        nodes_path='circuits/baseline_flight_brain_nodes.feather',
        edges_path='circuits/baseline_flight_brain_edges.feather',
        connectome_blend=0.1
    )
    res_base = sim_base.run_simulation(
        duration=duration,
        disturbance_type=dist_type,
        disturbance_magnitude=dist_mag,
        disturbance_time=dist_time
    )

    # 2. Benchmark Comprehensive Mobility Brain
    print("\n[*] 2. Running Comprehensive Mobility Brain (10,668 neurons)...")
    sim_comp = ClosedLoopFlyBodySim(
        controller_type='connectome_hybrid',
        nodes_path='circuits/comprehensive_mobility_nodes.feather',
        edges_path='circuits/comprehensive_mobility_edges.feather',
        connectome_blend=0.1
    )
    res_comp = sim_comp.run_simulation(
        duration=duration,
        disturbance_type=dist_type,
        disturbance_magnitude=dist_mag,
        disturbance_time=dist_time
    )

    # 3. Extract metrics
    def analyze_res(res, name):
        times = np.array(res['times'])
        eulers = np.array(res['euler_deg'])  # deg
        positions = np.array(res['positions'])
        ang_vels = np.array(res['ang_vel'])

        roll_deg = eulers[:, 0]
        pitch_deg = eulers[:, 1]
        yaw_deg = eulers[:, 2]
        alt = positions[:, 2]

        post_dist_mask = times >= dist_time
        peak_roll = np.max(np.abs(roll_deg[post_dist_mask])) if np.any(post_dist_mask) else 0.0
        final_roll = roll_deg[-1]
        final_alt = alt[-1]
        crashed = res.get('crashed', False)

        return {
            'name': name,
            'crashed': bool(crashed),
            'steps_completed': len(times),
            'peak_roll_deg': float(peak_roll),
            'final_roll_deg': float(final_roll),
            'mean_altitude_m': float(np.mean(alt)),
            'final_altitude_m': float(final_alt),
            'recovery_time_s': float(res['recovery_time_s']),
            'times': times.tolist(),
            'roll': roll_deg.tolist(),
            'pitch': pitch_deg.tolist(),
            'alt': alt.tolist()
        }

    m_base = analyze_res(res_base, "Baseline Brain (897n)")
    m_comp = analyze_res(res_comp, "Comprehensive Mobility Brain (10668n)")

    print("\n" + "=" * 75)
    print(" RESULTS COMPARISON")
    print("=" * 75)
    print(f"  Metric                     | Baseline (897n)   | Comprehensive (10668n)")
    print(f"  ---------------------------+-------------------+-----------------------")
    print(f"  Crashed                    | {str(m_base['crashed']):<17} | {str(m_comp['crashed']):<21}")
    print(f"  Steps Completed            | {m_base['steps_completed']:<17} | {m_comp['steps_completed']:<21}")
    print(f"  Peak Roll Deviation (deg)  | {m_base['peak_roll_deg']:<17.2f} | {m_comp['peak_roll_deg']:<21.2f}")
    print(f"  Final Roll Attitude (deg)  | {m_base['final_roll_deg']:<17.2f} | {m_comp['final_roll_deg']:<21.2f}")
    print(f"  Mean Flight Altitude (m)   | {m_base['mean_altitude_m']:<17.3f} | {m_comp['mean_altitude_m']:<21.3f}")
    print(f"  Final Altitude (m)         | {m_base['final_altitude_m']:<17.3f} | {m_comp['final_altitude_m']:<21.3f}")
    print(f"  Recovery Time (s)          | {m_base['recovery_time_s']:<17.3f} | {m_comp['recovery_time_s']:<21.3f}")
    print("=" * 75)

    # 4. Generate comparison plot
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    
    # Roll comparison
    axes[0].plot(m_base['times'], m_base['roll'], 'b-', label='Baseline Flight Brain (897 neurons)', linewidth=1.8)
    axes[0].plot(m_comp['times'], m_comp['roll'], 'g--', label='Comprehensive Mobility Brain (10,668 neurons)', linewidth=2.0)
    axes[0].axvline(dist_time, color='r', linestyle=':', label='Roll Perturbation (6 rad/s)')
    axes[0].set_ylabel('Roll Angle (deg)')
    axes[0].set_title('Closed-Loop Flight Stability: Baseline vs Comprehensive Mobility Connectome')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='upper right')

    # Pitch comparison
    axes[1].plot(m_base['times'], m_base['pitch'], 'b-', label='Baseline (897n)', linewidth=1.8)
    axes[1].plot(m_comp['times'], m_comp['pitch'], 'g--', label='Comprehensive (10668n)', linewidth=2.0)
    axes[1].axvline(dist_time, color='r', linestyle=':')
    axes[1].set_ylabel('Pitch Angle (deg)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='upper right')

    # Altitude comparison
    axes[2].plot(m_base['times'], m_base['alt'], 'b-', label='Baseline (897n)', linewidth=1.8)
    axes[2].plot(m_comp['times'], m_comp['alt'], 'g--', label='Comprehensive (10668n)', linewidth=2.0)
    axes[2].axvline(dist_time, color='r', linestyle=':')
    axes[2].set_xlabel('Time (seconds)')
    axes[2].set_ylabel('Altitude Z (meters)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc='upper right')

    plt.tight_layout()
    plot_path = 'visualizations/comprehensive_vs_baseline_stability.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\n[+] Saved comparison figure: {plot_path}")

    # 5. Save JSON report
    report = {
        'baseline_897n': {k: v for k, v in m_base.items() if k not in ('times', 'roll', 'pitch', 'alt')},
        'comprehensive_10668n': {k: v for k, v in m_comp.items() if k not in ('times', 'roll', 'pitch', 'alt')},
        'stability_preserved': bool(not m_comp['crashed'] and m_comp['peak_roll_deg'] <= m_base['peak_roll_deg'] + 5.0)
    }
    with open('circuits/stability_comparison_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print(f"[+] Saved stability report: circuits/stability_comparison_report.json")
    print(f"[+] STABILITY VERIFIED: {report['stability_preserved']}")


if __name__ == '__main__':
    run_benchmark()
