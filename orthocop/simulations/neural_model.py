"""
Brian2 Computational Neural Circuit Simulation for Drosophila Flight Connectome.
Implements:
1. Conductance-based Leaky Integrate-and-Fire (LIF) neuron model:
   C_m * dV/dt = -g_L * (V - E_L) + g_e * (E_e - V) + g_i * (E_i - V) + I_ext
   dg_e/dt = -g_e / tau_e
   dg_i/dt = -g_i / tau_i
2. Connectome-derived synaptic graph with neurotransmitter-specific excitation/inhibition.
3. Sensory current transduction (Haltere gyros, Vision, Wing strain).
4. Motor neuron spike decoding to wing kinematic control signals.
5. Neural metrics: Firing Rate, ISI, Response Latency, Fano Factor, Cross-Correlation.
6. Single and group neuron ablation support.
"""

import numpy as np
import pandas as pd
from scipy import signal
import brian2 as b2

# Configure Brian2 for fast standalone numpy codegen
b2.prefs.codegen.target = 'numpy'
b2.defaultclock.dt = 0.1 * b2.ms

class ConnectomeFlightNeuralModel:
    def __init__(self, nodes_df, edges_df, ablated_neurons=None, weight_scale=0.08 * b2.nS):
        self.nodes_df = nodes_df.copy().reset_index(drop=True)
        self.edges_df = edges_df.copy().reset_index(drop=True)
        self.ablated_neurons = set(ablated_neurons) if ablated_neurons else set()
        self.weight_scale = weight_scale
        
        # Filter out ablated neurons from edge list
        if self.ablated_neurons:
            self.edges_df = self.edges_df[
                (~self.edges_df['pre_id'].isin(self.ablated_neurons)) &
                (~self.edges_df['post_id'].isin(self.ablated_neurons))
            ].copy()
            
        # Map body_id to continuous integer index 0..N-1
        self.body_to_idx = {bid: idx for idx, bid in enumerate(self.nodes_df['body_id'])}
        self.idx_to_body = {idx: bid for bid, idx in self.body_to_idx.items()}
        self.N = len(self.nodes_df)
        
        # Categorize neuron indices by circuit role and effective side (somaSide or rootSide)
        self.roles = self.nodes_df['circuit_role'].values
        soma_sides = self.nodes_df['somaSide'].fillna('unknown').values
        root_sides = self.nodes_df['rootSide'].fillna('unknown').values if 'rootSide' in self.nodes_df.columns else soma_sides
        self.sides = [
            s if s in ['L', 'R', 'left', 'right'] else (r if r in ['L', 'R'] else 'unknown')
            for s, r in zip(soma_sides, root_sides)
        ]
        self.types = self.nodes_df['type'].values
        
        self.haltere_indices_L = [
            i for i, (r, s) in enumerate(zip(self.roles, self.sides))
            if r == 'haltere_sensory' and (s == 'L' or s == 'left')
        ]
        self.haltere_indices_R = [
            i for i, (r, s) in enumerate(zip(self.roles, self.sides))
            if r == 'haltere_sensory' and (s == 'R' or s == 'right')
        ]
        self.visual_indices_L = [
            i for i, (r, s) in enumerate(zip(self.roles, self.sides))
            if r == 'visual_sensory' and (s == 'L' or s == 'left')
        ]
        self.visual_indices_R = [
            i for i, (r, s) in enumerate(zip(self.roles, self.sides))
            if r == 'visual_sensory' and (s == 'R' or s == 'right')
        ]
        
        # Steering Motor Neurons (b1, b2, i1, i2) Left and Right
        self.mn_indices_L = [
            i for i, (r, s, t) in enumerate(zip(self.roles, self.sides, self.types))
            if r == 'motor_neuron' and (s == 'L' or s == 'left' or '_L' in t)
        ]
        self.mn_indices_R = [
            i for i, (r, s, t) in enumerate(zip(self.roles, self.sides, self.types))
            if r == 'motor_neuron' and (s == 'R' or s == 'right' or '_R' in t)
        ]
        
        # Build Brian2 network objects
        self.net = None
        self.neurons = None
        self.synapses_exc = None
        self.synapses_inh = None
        self.spike_mon = None
        self.state_mon = None
        self._build_brian_network()

    def _build_brian_network(self):
        """Constructs Brian2 LIF neurons and synaptic populations."""
        # Biophysical constants for Drosophila central neurons
        Cm = 200.0 * b2.pF          # Membrane capacitance
        gL = 10.0 * b2.nS           # Leak conductance (membrane time constant tau_m = Cm/gL = 20 ms)
        EL = -65.0 * b2.mV          # Resting potential
        Vth = -50.0 * b2.mV         # Spike threshold
        Vreset = -65.0 * b2.mV      # Reset potential
        tau_ref = 2.0 * b2.ms       # Refractory period
        
        Ee = 0.0 * b2.mV            # Excitatory reversal (ACh)
        Ei = -75.0 * b2.mV          # Inhibitory reversal (GABA/Glu)
        tau_e = 3.0 * b2.ms         # Excitatory decay constant
        tau_i = 6.0 * b2.ms         # Inhibitory decay constant

        self.namespace = {
            'Cm': Cm, 'gL': gL, 'EL': EL, 'Vth': Vth, 'Vreset': Vreset,
            'tau_ref': tau_ref, 'Ee': Ee, 'Ei': Ei, 'tau_e': tau_e, 'tau_i': tau_i
        }
        
        # Neuron equations
        eqs = """
        dv/dt = ( -gL*(v - EL) + ge*(Ee - v) + gi*(Ei - v) + I_ext ) / Cm : volt (unless refractory)
        dge/dt = -ge / tau_e : siemens
        dgi/dt = -gi / tau_i : siemens
        I_ext : amp
        """
        
        self.neurons = b2.NeuronGroup(
            self.N,
            eqs,
            threshold='v > Vth',
            reset='v = Vreset',
            refractory=tau_ref,
            method='euler',
            namespace=self.namespace
        )
        self.neurons.v = EL
        self.neurons.ge = 0.0 * b2.nS
        self.neurons.gi = 0.0 * b2.nS
        self.neurons.I_ext = 0.0 * b2.pA
        
        # Prepare synaptic edges partitioned by excitatory vs inhibitory transmitter
        exc_pre, exc_post, exc_w = [], [], []
        inh_pre, inh_post, inh_w = [], [], []
        
        # Neurotransmitter classification: acetylcholine/histamine/dopamine -> exc; gaba/glutamate -> inh
        nt_map = self.nodes_df.set_index('body_id')['consensus_nt'].to_dict()
        
        for _, row in self.edges_df.iterrows():
            u = int(row['pre_id'])
            v = int(row['post_id'])
            if u not in self.body_to_idx or v not in self.body_to_idx:
                continue
            idx_u = self.body_to_idx[u]
            idx_v = self.body_to_idx[v]
            
            w_raw = float(row['weight'])
            # Calibrated conductance weight: logarithmic scaling
            w_calib = float(self.weight_scale / b2.nS) * np.log1p(w_raw) * 1e-9 # in siemens
            
            nt = str(nt_map.get(u, 'acetylcholine')).lower()
            if 'gaba' in nt or 'glutamate' in nt:
                inh_pre.append(idx_u)
                inh_post.append(idx_v)
                inh_w.append(w_calib)
            else: # Default excitatory (cholinergic)
                exc_pre.append(idx_u)
                exc_post.append(idx_v)
                exc_w.append(w_calib)
                
        # Excitatory synapses
        if exc_pre:
            self.synapses_exc = b2.Synapses(
                self.neurons, self.neurons,
                model='w_syn : siemens',
                on_pre='ge_post += w_syn',
                delay=1.0 * b2.ms
            )
            self.synapses_exc.connect(i=exc_pre, j=exc_post)
            self.synapses_exc.w_syn = np.array(exc_w) * b2.siemens
            
        # Inhibitory synapses
        if inh_pre:
            self.synapses_inh = b2.Synapses(
                self.neurons, self.neurons,
                model='w_syn : siemens',
                on_pre='gi_post += w_syn',
                delay=1.2 * b2.ms
            )
            self.synapses_inh.connect(i=inh_pre, j=inh_post)
            self.synapses_inh.w_syn = np.array(inh_w) * b2.siemens

        # Monitors
        self.spike_mon = b2.SpikeMonitor(self.neurons)
        
        self.net = b2.Network()
        self.net.add(self.neurons)
        if self.synapses_exc:
            self.net.add(self.synapses_exc)
        if self.synapses_inh:
            self.net.add(self.synapses_inh)
        self.net.add(self.spike_mon)

    def set_sensory_inputs(self, sensory_dict):
        """Sets external injection currents I_ext for sensory populations."""
        # Reset all external inputs
        i_ext_arr = np.zeros(self.N)
        
        # Haltere Left and Right inputs
        i_h_L = sensory_dict.get('haltere_current_L', 0.0) # Amperes
        i_h_R = sensory_dict.get('haltere_current_R', 0.0)
        i_vis_L = sensory_dict.get('visual_current_L', 0.0)
        i_vis_R = sensory_dict.get('visual_current_R', 0.0)
        
        for idx in self.haltere_indices_L:
            i_ext_arr[idx] = i_h_L
        for idx in self.haltere_indices_R:
            i_ext_arr[idx] = i_h_R
        for idx in self.visual_indices_L:
            i_ext_arr[idx] = i_vis_L
        for idx in self.visual_indices_R:
            i_ext_arr[idx] = i_vis_R
            
        self.neurons.I_ext = i_ext_arr * b2.amp

    def step(self, duration=1.0 * b2.ms):
        """Advances Brian2 network simulation forward by duration."""
        self.net.run(duration, namespace=self.namespace)

    def decode_motor_commands(self, window_ms=10.0):
        """
        Decodes recent motor neuron firing rates into differential wing controls:
        d_A_L, d_A_R (differential stroke amplitude in radians)
        d_phi_0_L, d_phi_0_R (differential stroke offset in radians)
        """
        if self.spike_mon.num_spikes == 0:
            return {'d_A_L': 0.0, 'd_A_R': 0.0, 'd_phi_0_L': 0.0, 'd_phi_0_R': 0.0}
            
        current_time_ms = float(self.net.t / b2.ms)
        t_start_ms = max(0.0, current_time_ms - window_ms)
        
        spike_times_ms = np.array(self.spike_mon.t / b2.ms)
        spike_indices = np.array(self.spike_mon.i)
        
        # Recent spikes in window
        mask_recent = spike_times_ms >= t_start_ms
        recent_indices = spike_indices[mask_recent]
        
        # Count motor neuron spikes
        spikes_L = sum(np.isin(recent_indices, self.mn_indices_L))
        spikes_R = sum(np.isin(recent_indices, self.mn_indices_R))
        
        rate_L = spikes_L / (max(1, len(self.mn_indices_L)) * (window_ms * 1e-3)) # Hz
        rate_R = spikes_R / (max(1, len(self.mn_indices_R)) * (window_ms * 1e-3)) # Hz
        
        # Steering gain: differential rate of 100 Hz produces ~10 deg differential stroke amplitude
        gain_amplitude = np.deg2rad(0.10) # rad per Hz
        delta_rate = rate_L - rate_R
        
        # When Left MNs fire more, Left wing amplitude increases -> body rolls to the Right
        d_A_L = np.clip(delta_rate * gain_amplitude, -np.deg2rad(20.0), np.deg2rad(20.0))
        d_A_R = np.clip(-delta_rate * gain_amplitude, -np.deg2rad(20.0), np.deg2rad(20.0))
        
        return {
            'd_A_L': float(d_A_L),
            'd_A_R': float(d_A_R),
            'd_phi_0_L': 0.0,
            'd_phi_0_R': 0.0,
            'rate_mn_L': float(rate_L),
            'rate_mn_R': float(rate_R)
        }

    # Neural Metrics
    def compute_neural_statistics(self, duration_s=0.1):
        """Calculates Firing Rate, ISI, Latency, and Fano Factor from recorded spikes."""
        if self.spike_mon.num_spikes == 0:
            return {
                "total_spikes": 0,
                "active_neurons_count": 0,
                "mean_firing_rate_hz": 0.0,
                "mean_isi_ms": 0.0,
                "mean_fano_factor": 1.0
            }
            
        spike_times = np.array(self.spike_mon.t / b2.second)
        spike_indices = np.array(self.spike_mon.i)
        
        # 1. Neuronal firing rate: r = N_spikes / T
        rates = []
        fano_factors = []
        all_isis = []
        
        # Compute across neurons
        active_neurons = np.unique(spike_indices)
        for nid in active_neurons:
            ts = spike_times[spike_indices == nid]
            rate = len(ts) / duration_s
            rates.append(rate)
            
            # ISI: ISI_i = t_{i+1} - t_i
            if len(ts) > 1:
                isis = np.diff(ts)
                all_isis.extend(isis)
                
            # Fano Factor: F = Var(N) / E[N] over 20ms bins
            bin_size = 0.02 # 20 ms
            bins = np.arange(0, duration_s + bin_size, bin_size)
            counts, _ = np.histogram(ts, bins=bins)
            mean_c = np.mean(counts)
            var_c = np.var(counts)
            if mean_c > 0:
                fano_factors.append(var_c / mean_c)
                
        return {
            "total_spikes": int(self.spike_mon.num_spikes),
            "active_neurons_count": len(active_neurons),
            "mean_firing_rate_hz": float(np.mean(rates)) if rates else 0.0,
            "mean_isi_ms": float(np.mean(all_isis) * 1000.0) if all_isis else 0.0,
            "mean_fano_factor": float(np.mean(fano_factors)) if fano_factors else 1.0
        }

    @staticmethod
    def compute_cross_correlation(signal_x, signal_y):
        """Computes normalized cross-correlation: rho_xy(tau) = Cov(x, y(t+tau)) / (sigma_x * sigma_y)."""
        x = signal_x - np.mean(signal_x)
        y = signal_y - np.mean(signal_y)
        std_x = np.std(signal_x)
        std_y = np.std(signal_y)
        if std_x < 1e-9 or std_y < 1e-9:
            return np.zeros(len(signal_x))
        corr = signal.correlate(x, y, mode='same') / (std_x * std_y * len(x))
        return corr

if __name__ == "__main__":
    nodes = pd.read_feather("circuits/flight_candidate_neurons.feather")
    edges = pd.read_feather("circuits/flight_subnetwork_edges_min2.feather")
    
    print(f"[*] Initializing Brian2 model with {len(nodes):,} nodes and {len(edges):,} edges...")
    model = ConnectomeFlightNeuralModel(nodes, edges)
    
    print("[*] Running 20 ms open-loop stimulus test...")
    # Apply asymmetric sensory current (e.g. left roll disturbance)
    model.set_sensory_inputs({
        'haltere_current_L': 1.8e-9, # Left haltere stimulated
        'haltere_current_R': 1.0e-9  # Right haltere baseline
    })
    model.step(20.0 * b2.ms)
    
    stats = model.compute_neural_statistics(0.02)
    print("Neural test complete:")
    print("Total spikes recorded:", stats['total_spikes'])
    print("Active neurons:", stats['active_neurons_count'])
    print(f"Mean firing rate: {stats['mean_firing_rate_hz']:.1f} Hz")
    
    cmd = model.decode_motor_commands(window_ms=20.0)
    print("Decoded wing commands:", cmd)
