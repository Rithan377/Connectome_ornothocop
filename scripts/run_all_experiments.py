"""
Master Experiment Runner.
Executes the full experimental pipeline end-to-end:
1. Baseline Flight & Disturbance Rejection
2. Monte Carlo Statistical Evaluation
3. Parameter Sensitivity Analysis
4. Single & Group Neuron Ablation with FDR Control
5. Minimal Essential Connectome Synthesis (100% Control Retention)
6. Publication Figure Generation
"""

import os
import sys
import time

def main():
    print("================================================================================")
    print("  ORTHOCOP: MALE DROSOPHILA CONNECTOME ANALYSIS & FLIGHT SIMULATION PIPELINE")
    print("================================================================================")
    start_time = time.time()
    
    # 1. Baseline and Disturbance Rejection
    print("\n>>> STAGE 1: Baseline Flight & Disturbance Rejection Experiments")
    from orthocop.experiments.baseline_flight import run_baseline_and_disturbances
    run_baseline_and_disturbances()
    
    # 2. Monte Carlo Statistical Evaluation
    print("\n>>> STAGE 2: Monte Carlo Simulation & Statistical Uncertainty Quantification")
    from orthocop.experiments.monte_carlo import run_monte_carlo_experiments
    run_monte_carlo_experiments(n_trials=30)
    
    # 3. Parameter Sensitivity Analysis
    print("\n>>> STAGE 3: Finite-Difference Parameter Sensitivity Analysis")
    from orthocop.experiments.sensitivity_analysis import run_sensitivity_analysis
    run_sensitivity_analysis()
    
    # 4. Neuron and Group Ablation Analysis
    print("\n>>> STAGE 4: Neuron & Synaptic Ablation Analysis (FDR Control)")
    from orthocop.experiments.ablation_analysis import run_ablation_experiments
    run_ablation_experiments()
    
    # 5. Minimal Essential Flight Connectome Synthesis
    print("\n>>> STAGE 5: Synthesis of Minimal Essential Connectome (100% Control Retention)")
    from orthocop.experiments.minimal_circuit_synthesis import MinimalFlightCircuitSynthesizer
    synthesizer = MinimalFlightCircuitSynthesizer()
    synthesizer.synthesize_minimal_circuit()
    
    # 6. Visualization Generation
    print("\n>>> STAGE 6: Generating Publication-Grade Visualizations")
    from orthocop.visualization.plot_results import generate_all_visualizations
    generate_all_visualizations()
    
    elapsed = time.time() - start_time
    print("\n================================================================================")
    print(f"  [+] END-TO-END PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.1f} SECONDS")
    print("================================================================================")

if __name__ == "__main__":
    main()
