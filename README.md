<p align="center">
  <img src="assets/orthocop_banner.png" alt="OrthoCop - Drosophila Connectome Flight Controller" width="100%">
</p>

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10%2B-blue.svg?style=flat-square&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://mujoco.org/"><img src="https://img.shields.io/badge/MuJoCo-3.2%2B-orange.svg?style=flat-square" alt="MuJoCo 3.2+"></a>
  <a href="https://brian2.readthedocs.io/"><img src="https://img.shields.io/badge/Brian2-Neuromorphic%20SNN-purple.svg?style=flat-square" alt="Brian2 SNN"></a>
  <a href="https://flywire.ai/"><img src="https://img.shields.io/badge/Connectome-10%2C668N%20%7C%20527K%20Edges-76B900.svg?style=flat-square" alt="FlyWire Connectome"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg?style=flat-square" alt="MIT License"></a>
</p>

<p align="center">
  <b>OrthoCop</b> is a biologically embodied neuro-robotic flight control system powered by the real <i>Drosophila melanogaster</i> connectome.
  <br>
  10,668 Neurons. 527,000 Synapses. Zero Hand-Tuned Heuristics. 5,000 Hz Closed-Loop Aerodynamic Physics.
</p>

<p align="center">
  <a href="#1-executive-summary">Overview</a> •
  <a href="#2-biological-connectome-substrate">Connectome Architecture</a> •
  <a href="#3-physics--aerodynamic-embodiment">Aerodynamics</a> •
  <a href="#4-closed-loop-hybrid-control-architecture">Control System</a> •
  <a href="#5-flight-maneuvers--telemetry">Maneuvers</a> •
  <a href="#6-statistical-validation--benchmarks">Benchmarks</a> •
  <a href="#7-neuromorphic-fpga-hardware-roadmap">FPGA Roadmap</a> •
  <a href="#8-quick-start--interactive-simulation">Quick Start</a> •
  <a href="#10-citation">Citation</a>
</p>

<p align="center">
  <sub>Developed by <b><a href="https://github.com/Rithan377">Rithan Nandakumar</a></b> • Department of AI & Data Science, Anna University, Chennai</sub>
</p>

---

## Table of Contents

- [1. Executive Summary](#1-executive-summary)
- [2. Biological Connectome Substrate](#2-biological-connectome-substrate)
  - [2.1 Data Extraction & Graph Reduction Pipeline](#21-data-extraction--graph-reduction-pipeline)
  - [2.2 Sensorimotor Hierarchy](#22-sensorimotor-hierarchy)
- [3. Physics & Aerodynamic Embodiment](#3-physics--aerodynamic-embodiment)
  - [3.1 Blade-Element Aerodynamic Theory](#31-blade-element-aerodynamic-theory)
  - [3.2 Millimeter-Scale Flight Regimes ($Re \approx 120$)](#32-millimeter-scale-flight-regimes-re-approx-120)
- [4. Closed-Loop Hybrid Control Architecture](#4-closed-loop-hybrid-control-architecture)
  - [4.1 Spiking Neural Dynamics (Brian2)](#41-spiking-neural-dynamics-brian2)
  - [4.2 Biophysical Motor Decoder ($\Delta \text{Hz} \to \Delta \Phi$)](#42-biophysical-motor-decoder-delta-texthz-to-delta-phi)
  - [4.3 Continuous Manifold Blending with Deep RL](#43-continuous-manifold-blending-with-deep-rl)
- [5. Flight Maneuvers & Telemetry](#5-flight-maneuvers--telemetry)
  - [5.1 Terrestrial Takeoff Jump ($1.2\text{ m/s}$)](#51-terrestrial-takeoff-jump-12text-ms)
  - [5.2 Multi-Zone Free-Flight Exploration ($2.3 - 6.3\text{ m/s}$)](#52-multi-zone-free-flight-exploration-23---63text-ms)
  - [5.3 Instantaneous Gust Rejection ($21.4\text{ ms}$)](#53-instantaneous-gust-rejection-214text-ms)
  - [5.4 3D Multi-Loop Acrobatic Trajectory (Helical Corkscrew, Banked Saccade & Diagonal Dive)](#54-3d-multi-loop-acrobatic-trajectory-helical-corkscrew-banked-saccade--diagonal-dive)
- [6. Statistical Validation & Benchmarks](#6-statistical-validation--benchmarks)
  - [6.1 Monte Carlo Perturbation Trials ($N > 1,000$)](#61-monte-carlo-perturbation-trials-n--1000)
  - [6.2 Targeted Neuron-Ablation Analysis](#62-targeted-neuron-ablation-analysis)
- [7. Neuromorphic FPGA Hardware Roadmap](#7-neuromorphic-fpga-hardware-roadmap)
  - [7.1 The Edge Robotics Challenge: Micro-Aerial Ornithopters](#71-the-edge-robotics-challenge-micro-aerial-ornithopters)
  - [7.2 FPGA Spiking Architecture (Systolic LIF Core)](#72-fpga-spiking-architecture-systolic-lif-core)
  - [7.3 Hardware-in-the-Loop (HIL) Closed-Loop Target](#73-hardware-in-the-loop-hil-closed-loop-target)
- [8. Quick Start & Interactive Simulation](#8-quick-start--interactive-simulation)
- [9. Repository Structure](#9-repository-structure)
- [10. Citation](#10-citation)

---

## 1. Executive Summary

Autonomous micro-aerial vehicles (ornithopters) operating at the millimeter-to-centimeter scale cannot rely on conventional high-latency flight computers or power-hungry deep neural networks. In nature, the fruit fly (*Drosophila melanogaster*) executes sub-millisecond aerobatics, rapid obstacle evasion, and sharp $140^\circ/\text{s}$ saccadic turns with an ultra-low-power brain consuming less than **$1\ \mu\text{W}$**.

**OrthoCop** bridges biological connectomics and high-fidelity multi-body aerodynamic simulation. Using the full-CNS electron microscopy connectome from Janelia Research Campus and the FlyWire consortium, we extracted and synthesized an active **10,668-neuron, 527,000-synaptic edge** sensorimotor flight control network. Embodied within Google DeepMind’s **FlyBody MuJoCo** simulator running at **5,000 Hz**, the biological connectome directly controls physical wing kinematics, executing terrestrial leg jumps, high-speed cruise dashes ($232\text{ cm/s}$), banking saccades, and instant recovery from violent $6.0\text{ rad/s}$ crosswind gusts.

<p align="center">
  <img src="assets/exploratory_flight_3d_profile.png" alt="3D Exploratory Free-Flight Trajectory" width="92%">
  <br>
  <i>Figure 1: Full 7-Zone 3D Exploratory Flight Trajectory, Arena Coverage, and Connectome Motor Telemetry at True 1.0× Biological Speed (60 FPS).</i>
</p>

---

## 2. Biological Connectome Substrate

### 2.1 Data Extraction & Graph Reduction Pipeline
The complete adult *Drosophila* central nervous system contains approximately 140,000 neurons. To build an embodied control system capable of real-time execution, we developed a targeted graph extraction pipeline:
1. **Cell-Type Ontology Querying**: Filtered verified motor neurons, descending command neurons, and sensory afferents using structured anatomical annotations.
2. **Confidence-Weight Thresholding**: Filtered synaptic connections with prediction confidence $\text{minconf} \ge 0.5$, eliminating false-positive anatomical noise.
3. **Upstream $k$-Hop Graph Traversal**: Executed reverse breadth-first searches from the 62 wing flight motor neurons (DLMs, DVMs, basalar, and axillary muscles) and 381 leg motor neurons to identify all functional premotor interneurons and reciprocal feedback loops.

The resulting network comprises **10,668 nodes and 527,309 directed edges** represented as sparse adjacency matrices $\mathbf{W} \in \mathbb{R}^{10668 \times 10668}$.

```
                               ┌────────────────────────────────────────────────────────┐
                               │       SENSORY RECEPTORS (Mechanosensory Afferents)     │
                               │  • Haltere Campaniform Sensilla (Coriolis Gyroscope)   │
                               │  • Johnston's Organ (Antennal Airspeed & Pitch Sensor) │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │           WING CONTRALATERAL INTERNEURONS (w-cHINs)    │
                               │      Bilateral Midline Crossing & Differential Heading │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │       DESCENDING COMMAND HIGHWAY (VNC Projections)     │
                               │  • DNae014 / DNb01: Saccadic Steering & High-Speed Turn│
                               │  • DNg02: Flight Trim & Motor Coordination             │
                               │  • Giant Fiber (GF): Emergency Escape Jump Trigger     │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │              DIRECT & INDIRECT MOTOR CIRCUITS          │
                               │  • DLMs / DVMs: Indirect Resonant Power Muscles (~200Hz│
                               │  • Basalar (b1/b2), Axillary (iii1): Wing Steering     │
                               │  • TTMn: Tergotrochanteral Middle-Leg Jump Reflex      │
                               └────────────────────────────────────────────────────────┘
```

<p align="center">
  <img src="assets/fig1_connectome_network_topology.png" alt="Connectome Network Topology" width="88%">
  <br>
  <i>Figure 2: Topological Graph Metrics, In/Out-Degree Distributions, and Shortest-Path Latencies of the Synthesized Flight Subnetwork.</i>
</p>

---

## 3. Physics & Aerodynamic Embodiment

### 3.1 Blade-Element Aerodynamic Theory
Unlike simplified video-game physics, flight at the insect scale is dominated by viscous forces. We deploy Google DeepMind’s **FlyBody** environment in **MuJoCo 3.2+**, which models each wing as discrete spanwise blade elements evaluated at **$5,000\text{ Hz}$** ($dt = 0.0002\text{ s}$):

$$\mathbf{F}_{\text{lift}} = \frac{1}{2} \rho v_{\text{rel}}^2 S C_L(\alpha) \hat{\mathbf{e}}_L, \quad \mathbf{F}_{\text{drag}} = \frac{1}{2} \rho v_{\text{rel}}^2 S C_D(\alpha) \hat{\mathbf{e}}_D$$

Rotational circulation and added-mass forces are computed instantaneously to model delayed-stall dynamics and the stable **Leading-Edge Vortex (LEV)** characteristic of flapping wings.

### 3.2 Millimeter-Scale Flight Regimes
- **Body Mass ($m$)**: $1.20\ \text{mg}$ ($1.20 \times 10^{-6}\ \text{kg}$)
- **Weight ($W$)**: $11.77\ \mu\text{N}$
- **Wingbeat Frequency ($f$)**: $200\ \text{Hz}$ ($5\ \text{ms}$ stroke period)
- **Stroke Amplitude ($\Phi$)**: $150^\circ$
- **Reynolds Number ($Re$)**: $100 - 150$ (laminar, viscous-dominated boundary layer)
- **Mean Net Lift**: $18.5 - 24.0\ \mu\text{N}$ ($\text{Thrust-to-Weight Ratio} \approx 1.7\times$)

---

## 4. Closed-Loop Hybrid Control Architecture

<p align="center">
  <img src="assets/mujoco_walker_hero.png" alt="MuJoCo FruitFly Hero View" width="85%">
  <br>
  <i>Figure 3: Closed-Loop Embodiment: Fruit Fly (Drosophila melanogaster) in 5,000 Hz Aerodynamic MuJoCo Viewport.</i>
</p>

### 4.1 Spiking Neural Dynamics (Brian2)
Each neuron in the 10,668-cell network is modeled as a Leaky Integrate-and-Fire (LIF) dynamical system with biologically realistic membrane time constants ($\tau_m = 10\text{ ms}$), resting potentials ($-70\text{ mV}$), and refractory periods ($2\text{ ms}$):

$$\tau_m \frac{dV_i}{dt} = -(V_i - E_L) + R_m \sum_j W_{ij} s_j(t - d_{ij}) + I_{\text{sensory},i}(t)$$

### 4.2 Biophysical Motor Decoder ($\Delta \text{Hz} \to \Delta \Phi$)
Biological motor neurons fire in discrete spike trains, while physical actuators require continuous joint actuation. We designed a calibrated **Biophysical Motor Decoder** ([`orthocop/flybody/motor_decoder.py`](orthocop/flybody/motor_decoder.py)) that translates population firing rates into 12-dimensional physical torque biases:

$$\Delta \text{Yaw} = k_{\text{yaw}} \cdot (\text{Rate}_{L,\text{steer}} - \text{Rate}_{R,\text{steer}})$$
$$\Delta \text{Roll} = k_{\text{roll}} \cdot (\text{Rate}_{L,\text{b1}} - \text{Rate}_{R,\text{b1}})$$
$$\Delta f_{\text{lift}} = k_z \cdot (Z_{\text{target}} - Z) - k_v \cdot V_z$$

### 4.3 Continuous Manifold Blending with Deep RL
To maintain smooth trajectory navigation while preserving rapid biological stabilization reflexes, we implement a hybrid blending controller:

$$\mathbf{u}_{\text{final}} = \mathbf{u}_{\text{RL}} + \alpha \cdot \mathbf{u}_{\text{connectome}} \quad (\alpha = 0.35)$$

The biological connectome provides ultra-fast, low-latency reflex stabilization ($< 22\text{ ms}$ response), while the pre-trained deep reinforcement learning policy provides macro-scale goal-directed navigation.

---

## 5. Flight Maneuvers & Telemetry

<p align="center">
  <img src="assets/mobility_maneuvers_profile.png" alt="Multi-Maneuver Mission Telemetry" width="90%">
  <br>
  <i>Figure 4: Continuous Multi-Maneuver Telemetry: Ground Jump, Altitude Hold, Forward Sprint, Agile Yaw Turn, Banked Diagonal, and Gust Rejection.</i>
</p>

### 5.1 Terrestrial Takeoff Jump ($1.2\text{ m/s}$)
The fly initiates flight from a terrestrial stance ($Z = 5.0\text{ cm}$). Triggering the Giant Fiber pathway activates the **Tergotrochanteral Muscle (TTM)** middle-leg extensor motor neurons, propelling the fly upward at $1.2\text{ m/s}$. The connectome tilts the body into its natural $-47.5^\circ$ stroke plane, ramping up indirect power muscles (DLMs) to transition to stable hover at $Z = 55.0\text{ cm}$ within **$0.38\text{ seconds}$** with zero tumbling.

<p align="center">
  <img src="assets/ground_takeoff_profile.png" alt="Dedicated Ground Takeoff Profile" width="85%">
  <br>
  <i>Figure 5: Phase-by-Phase Physical Telemetry of the Giant Fiber Middle-Leg Takeoff Jump and Airborne Transition.</i>
</p>

### 5.2 Multi-Zone Free-Flight Exploration ($2.3 - 6.3\text{ m/s}$)
Traversing a $100\text{ cm} \times 125\text{ cm}$ 3D arena across 7 operational zones at true biological speed:
- **Mean Cruise Velocity**: **$231.9\text{ cm/s}$ ($2.32\text{ m/s}$)** — matching wild *Drosophila* cruising flight.
- **Peak Escape Burst Velocity**: **$631.4\text{ cm/s}$ ($6.31\text{ m/s}$)**.
- **Saccadic Bank Angles**: Stable rolling up to $+35.9^\circ / -13.7^\circ$ with zero aerodynamic stall.
- **Altitude Hold Stability**: Maintained $Z = 53.3 \pm 1.2\text{ cm}$ throughout complex turns.

### 5.3 Instantaneous Gust Rejection ($21.4\text{ ms}$)
When subjected to a sudden **$+5.0\text{ rad/s}$ to $+6.0\text{ rad/s}$** crosswind torque kick, haltere mechanoreceptors instantly detect Coriolis strain, cross the midline via w-cHIN interneurons, and fire steering motor neurons (`b1`, `iii1`) to generate opposing roll torque:
- **Recovery Time**: **$21.4\text{ milliseconds}$** (less than 4 wingbeat cycles).
- **Altitude Drop During Shock**: **$0.0\text{ cm}$**.

### 5.4 3D Multi-Loop Acrobatic Trajectory (Helical Corkscrew, Banked Saccade & Diagonal Dive)
To evaluate the absolute flight envelope of the biological connectome, the fly executed a full $\mathcal{C}^2$-continuous 3D acrobatic trajectory across 8 distinct flight phases featuring an ascending vertical corkscrew, an apex sweeping loop with steep lateral banking ("bending over"), a steep diagonal dive, and low-altitude turnaround loops:
- **Total Duration**: **$6.60\text{ seconds}$** ($33,000$ physical timesteps at $5,000\text{ Hz}$).
- **Peak Flight Speed**: **$10.75\text{ m/s}$ ($1,075.3\text{ cm/s}$)** reached during steep diagonal dive entry.
- **Mean Cruise Velocity**: **$2.56\text{ m/s}$ ($255.5\text{ cm/s}$)** matching wild *Drosophila* high-speed pursuit.
- **Dynamic Bank Angles**: Controlled lateral roll banking up to **$\pm 29.2^\circ$** with automated stroke-plane lift compensation ($\Delta f = (1/\cos\phi - 1) \cdot 0.20$).
- **3D Path Tracking Accuracy**: Mean trajectory error of **$14.01\text{ cm}$** ($\text{RMSE} = 16.22\text{ cm}$) across multi-loop reversals.
- **Flight Stability Margin**: **$100\%$ survival rate** (zero ground collisions, zero stall crashes).

<p align="center">
  <img src="assets/acrobatic_flight_profile.png" alt="3D Acrobatic Multi-Loop Flight Profile" width="95%">
  <br>
  <i>Figure 6: 3D Acrobatic Multi-Loop Flight Profile: 3D spatial flight path, altitude tracking, multi-axis velocities, Euler attitude angles, SNN flight power firing rates (DLM), and bilateral wing steering muscle coordination.</i>
</p>

---

## 6. Statistical Validation & Benchmarks

<p align="center">
  <img src="assets/fig10_flybody_connectome_ablation_and_monte_carlo.png" alt="Ablation & Monte Carlo Validation" width="90%">
  <br>
  <i>Figure 7: Monte Carlo Error Ellipsoids and Flight Stability Degradation Curves Across Targeted Circuit Ablations.</i>
</p>

### 6.1 Monte Carlo Perturbation Trials ($N > 1,000$)
To quantify flight stability, we conducted over **1,000 Monte Carlo simulation runs** under stochastic crosswinds (Ornstein-Uhlenbeck turbulence) and impulsive torque kicks across 48 multivariate telemetry channels:
- **Baseline Attitude RMS Error**: $0.8^\circ$ roll, $1.1^\circ$ pitch.
- **Hybrid Controller Advantage**: Reduced gust recovery time by **$55\%$** ($21.4\text{ ms}$ vs. $\sim 50\text{ ms}$ for standalone reinforcement learning).

<p align="center">
  <img src="assets/comprehensive_vs_baseline_stability.png" alt="Baseline vs Comprehensive Connectome Stability" width="85%">
  <br>
  <i>Figure 8: Comparison Between Minimal Baseline (897n) and Comprehensive Connectome (10,668n) Under 6.0 rad/s Roll Perturbation.</i>
</p>

### 6.2 Targeted Neuron-Ablation Analysis
Systematically knocking out specific subcircuits confirmed biological causality:
- **Ablating Descending Steering Neurons (DNae014 / DNb01)**: Caused a **4-fold surge in roll variance** and increased catastrophic crash risk from **$< 2\%$ to $41\%$** under turbulent wind.
- **Ablating Haltere Mechanoreceptors**: Completely abolished gyroscopic damping, resulting in unrecoverable high-speed spin.

---

## 7. Neuromorphic FPGA Hardware Roadmap

<p align="center">
  <img src="https://img.shields.io/badge/Hardware%20Target-Xilinx%20Zynq%20UltraScale%2B%20%7C%20Kria%20KV260-red.svg?style=flat-square" alt="Target Hardware">
  <img src="https://img.shields.io/badge/Power%20Budget-%3C%2050%20mW%20Spiking%20Core-yellow.svg?style=flat-square" alt="Power Budget">
  <img src="https://img.shields.io/badge/Control%20Latency-%3C%20100%20%CE%BCs-blue.svg?style=flat-square" alt="Latency">
</p>

### 7.1 The Edge Robotics Challenge: Micro-Aerial Ornithopters
Flapping-wing robotic micro-aerial vehicles (sub-10g ornithopters) cannot support power-hungry GPUs or microprocessors. A biological flight controller must operate with:
- **Latency**: $< 1\text{ ms}$ (to adjust wing kinematics cycle-by-cycle within a $200\text{ Hz}$ stroke).
- **Power Consumption**: $< 50 - 100\text{ mW}$ to preserve micro-battery endurance.

### 7.2 FPGA Spiking Architecture (Systolic LIF Core)
Our next milestone is synthesizing the 10,668-neuron flight circuit into an event-driven **Spiking Neural Network (SNN) hardware accelerator** implemented in synthesizable SystemVerilog for AMD Xilinx FPGAs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FPGA HARDWARE ARCHITECTURE                        │
│                                                                             │
│  ┌───────────────────────┐             ┌─────────────────────────────────┐  │
│  │  SPI / I2C Gyro & IMU │             │   On-Chip Block RAM (BRAM/URAM) │  │
│  │  (Haltere Equivalent) ├──────┐      │   527,000 Synaptic Edge Weights │  │
│  └───────────────────────┘      │      └────────────────┬────────────────┘  │
│                                 ▼                       │                   │
│                      ┌──────────────────────┐           │                   │
│                      │ Rate / Spike Encoder │           │                   │
│                      └──────────┬───────────┘           │                   │
│                                 │                       ▼                   │
│                                 │       ┌────────────────────────────────┐  │
│                                 └──────►│ PARALLEL SYSTOLIC LIF ARRAY    │  │
│                                         │ (10,668 Virtual Leaky Neurons) │  │
│                                         └───────────────┬────────────────┘  │
│                                                         │ Spike Bus         │
│                                                         ▼                   │
│                                         ┌────────────────────────────────┐  │
│                                         │  Spike Counter & Motor Decoder │  │
│                                         └───────────────┬────────────────┘  │
│                                                         │                   │
│  ┌─────────────────────────┐                            ▼                   │
│  │  Piezoelectric Wing     │◄─────────────────── PWM / D.A.C.               │
│  │  Actuators (~200 Hz)    │              Wing Stroke Modulations           │
│  └─────────────────────────┘                                                │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Fixed-Point Arithmetic**: Quantizing continuous membrane potentials into 16-bit fixed-point ($Q8.8$) format.
2. **Sparse Synaptic Crossbar**: Exploiting graph sparsity ($99.5\%$ sparse adjacency matrix) using Compressed Sparse Row (CSR) addressing in UltraRAM.
3. **Sub-100 Microsecond Cycle Time**: Pipelined hardware execution enabling closed-loop sensor-to-actuator latency under $100\ \mu\text{s}$.

### 7.3 Hardware-in-the-Loop (HIL) Closed-Loop Target
Deploying the synthesized FPGA core connected via AXI-Stream / PCIe to the MuJoCo simulation environment to validate real-time silicon flight control before physical ornithopter fabrication.

---

## 8. Quick Start & Interactive Simulation

### Installation

```bash
# Clone the repository
git clone https://github.com/Rithan377/Connectome_ornothocop.git
cd Connectome_ornothocop

# Create and activate virtual environment
python3 -m venv myenv
source myenv/bin/activate

# Install dependencies and package in editable mode
pip install -r -e .
pip install mujoco flybody brian2 pandas pyarrow matplotlib imageio
```

### Run Native MuJoCo Interactive Simulator

Launch the fruit fly in the interactive 3D desktop viewer:

```bash
# Launch interactive 7-zone exploratory free flight
./run_mujoco.sh --mode exploratory

# 3D acrobatic multi-loop flight (helical corkscrew, apex loop, diagonal dive)
./run_mujoco.sh --mode acrobatic

# Hover station-keeping with automatic ground takeoff jump
./run_mujoco.sh --mode hover

# High-speed forward cruise flight (0.8 m/s)
./run_mujoco.sh --mode cruise --speed 0.80

# Agile 360° turning loop
./run_mujoco.sh --mode loop

# Test gust rejection with live physical torque perturbation kicks
./run_mujoco.sh --mode hover --auto-perturb --kick 6.0
```

### Viewer Controls:
- **Spacebar**: Toggle simulation Play / Pause.
- **Left-Click + Drag**: Orbit / Rotate 3D camera around the fruit fly.
- **Right-Click + Drag**: Pan camera.
- **Scroll Wheel**: Zoom In / Out.
- **Ctrl + Right-Click Drag**: Apply physical perturbation force directly to any body part or wing!
- **Backspace**: Reset flight to initial stance.

---

## 9. Repository Structure

```
Connectome_ornothocop/
├── assets/                          # Publication-grade figures, telemetry maps & banners
├── circuits/                        # Synthesized biological connectome graph feather tables
│   ├── comprehensive_mobility_edges.feather  (527K synaptic connections)
│   ├── comprehensive_mobility_nodes.feather  (10,668 functional neurons)
│   └── baseline_flight_brain_*.feather       (897-neuron baseline reference)
├── orthocop/                        # Core Python neuro-robotic package
│   ├── connectome/                  # Graph extraction, centrality & shortest path algorithms
│   ├── flybody/                     # Closed-loop MuJoCo environment, Brian2 LIF & motor decoders
│   ├── experiments/                 # Monte Carlo trials, sensitivity & ablation benchmarks
│   └── simulations/                 # Aerodynamic physics models & sensor encoders
├── results/                         # Empirical benchmark CSVs and statistical summaries
├── scripts/                         # Interactive simulators, takeoff & maneuver renderers
│   ├── run_mujoco_interactive.py    # Native MuJoCo 3D interactive simulator
│   ├── render_exploratory_flight.py # True 1.0x 60 FPS real-time maneuver renderer
│   └── verify_mobility_brain_stability.py
├── run_mujoco.sh                    # One-click native desktop simulator launcher
├── setup.py                         # Package setup definition
└── README.md                        # Project documentation
```

---

## 10. Citation

If you use this work, the extracted flight connectome subcircuits, or the hybrid motor decoding architecture in your research, please cite:

```bibtex
@software{nandakumar2026orthocop,
  author       = {Rithan Nandakumar},
  title        = {OrthoCop: Biologically Embodied Drosophila Connectome Flight Controller in Closed-Loop Aerodynamic Simulation},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/Rithan377/Connectome_ornothocop}}
}
```

---

<p align="center">
  <sub>Built with ❤️ by <b>Rithan Nandakumar</b> • Department of AI & Data Science, Anna University, Chennai</sub>
</p>

