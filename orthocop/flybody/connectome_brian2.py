"""
Connectome Spiking Neural Network Engine for Flight Control.

Implements biophysical Leaky Integrate-and-Fire (LIF) network with conductance-based
synapses based on the Male Drosophila CNS Connectome (HHMI Janelia / Google Research).
Supports:
- Stabilized flight circuit (897 neurons, 34,011 synapses, 300 inhibitory premotor INs)
- Full motor connectome (10,663 neurons, 527,218 synapses) via scipy.sparse
- Ultra-minimal circuit (148 neurons, 1,095 synapses)
- All 23 wing motor neuron types grouped into roll/pitch/yaw functional pools
- Bilateral L/R partitioning via somaSide coordinates
- Structural graph analysis with NetworkX
- Targeted neuron, population, and synaptic ablation experiments
"""

import os
from typing import Dict, Any, List, Optional, Set
import numpy as np
import pandas as pd
import networkx as nx
import pyarrow.feather as feather

try:
    from scipy import sparse as sp
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class ConnectomeFlightCircuit:
    """
    Connectome-derived Spiking Neural Network for Drosophila flight stabilization.

    Uses scipy.sparse CSR matrices for circuits > 500 neurons.
    Groups all 23 wing motor neuron types into 3 functional actuator pools:
      - Roll: b2, tp1, tp2, tpn (stroke plane angle / deviation)
      - Pitch: b1, hg1-4, hiii2, iii1 (angle of attack / feathering)
      - Yaw: i1, i2, ps1, ps2, hi1, hi2 (stroke amplitude / pronation)
    Plus bilateral power pools: DLM_L/R, DVM_L/R, hDVM
    """

    def __init__(
        self,
        nodes_path: str = 'circuits/stabilized_flight_nodes.feather',
        edges_path: str = 'circuits/stabilized_flight_edges.feather',
        dt: float = 0.0001,          # Neural simulation timestep (0.1 ms)
        step_dt: float = 0.001,      # Interface step timestep (1.0 ms)
        ablated_nodes: Optional[Set[int]] = None,
        ablate_halteres: bool = False,
        ablate_visual: bool = False
    ):
        self.dt = dt
        self.step_dt = step_dt
        self.steps_per_call = max(1, int(round(step_dt / dt)))

        self.ablated_nodes = set(ablated_nodes) if ablated_nodes else set()
        self.ablate_halteres = ablate_halteres
        self.ablate_visual = ablate_visual

        # 1. Load nodes and edges
        self.nodes_df = feather.read_table(nodes_path).to_pandas()
        self.edges_df = feather.read_table(edges_path).to_pandas()

        self.num_neurons = len(self.nodes_df)
        self.body_ids = self.nodes_df['body_id'].tolist()
        self.id_to_idx = {bid: idx for idx, bid in enumerate(self.body_ids)}

        # Use sparse matrices for large circuits (> 500 neurons)
        self.use_sparse = HAS_SCIPY and self.num_neurons > 500

        # 2. Build NetworkX structural graph (vectorized fast edge loading)
        self.graph = nx.DiGraph()
        for row in self.nodes_df.itertuples():
            self.graph.add_node(
                int(row.body_id),
                subclass=getattr(row, 'subclass', ''),
                type=getattr(row, 'type', ''),
                circuit_role=getattr(row, 'circuit_role', ''),
                side=getattr(row, 'side', ''),
                consensus_nt=getattr(row, 'consensus_nt', '')
            )
        if len(self.edges_df) > 0:
            pre_arr = self.edges_df['pre_id'].values
            post_arr = self.edges_df['post_id'].values
            w_arr = self.edges_df['weight'].values
            for p, q, w in zip(pre_arr, post_arr, w_arr):
                if p in self.id_to_idx and q in self.id_to_idx:
                    self.graph.add_edge(int(p), int(q), weight=float(w))

        # 3. Categorize neuron indices by modality and side
        self._categorize_neurons()

        # 4. Construct synaptic adjacency matrices (Excitatory ACh and Inhibitory GABA/Glu)
        self._build_synaptic_matrices()

        # 5. Initialize neural state variables
        self.reset()

    def _resolve_side(self, side_val, soma_side, root_side, inst):
        """
        Resolve neuron hemisphere (L/R) using all available metadata.
        Priority: side column > somaSide > rootSide > instance suffix > default L.
        """
        side = str(side_val or '')
        if side in ('L', 'R'):
            return side

        soma = str(soma_side or '')
        if soma in ('L', 'left'):
            return 'L'
        if soma in ('R', 'right'):
            return 'R'

        root = str(root_side or '')
        if root in ('L', 'left'):
            return 'L'
        if root in ('R', 'right'):
            return 'R'

        inst_s = str(inst or '')
        if inst_s.endswith('_L'):
            return 'L'
        if inst_s.endswith('_R'):
            return 'R'

        if soma == 'M':
            return 'M'

        return 'L'

    def _categorize_neurons(self):
        """
        Map neuron body IDs to functional pools using all available metadata.
        Groups all 23 wing motor neuron types into roll/pitch/yaw actuator pools.
        """
        # Sensory pools
        self.haltere_L_idx = []
        self.haltere_R_idx = []
        self.visual_L_idx = []
        self.visual_R_idx = []
        self.wing_L_idx = []
        self.wing_R_idx = []

        # Grouped flight steering motor pools (multiple MN types per pool)
        # Roll group: b2, tp1, tp2, tpn (stroke plane / deviation angle)
        self.mn_roll_L = []
        self.mn_roll_R = []
        # Pitch group: b1, hg1-4, hiii2, iii1 (angle of attack / feathering)
        self.mn_pitch_L = []
        self.mn_pitch_R = []
        # Yaw group: i1, i2, ps1, ps2, hi1, hi2 (stroke amplitude bias)
        self.mn_yaw_L = []
        self.mn_yaw_R = []

        # Bilateral power motor pools (asynchronous muscles)
        self.mn_dlm_L = []
        self.mn_dlm_R = []
        self.mn_dvm_L = []
        self.mn_dvm_R = []
        self.mn_hdvm = []

        # Abdominal motor pools
        self.mn_abd_L = []
        self.mn_abd_R = []
        self.mn_abd_pitch = []

        # Neck motor pools (gaze stabilization)
        self.mn_neck_L = []
        self.mn_neck_R = []

        # Haltere motor pools (active tuning)
        self.mn_hm_L = []
        self.mn_hm_R = []

        # High-power jump reflex pool (Tergotrochanteral muscle TTMn)
        self.mn_ttm_L = []
        self.mn_ttm_R = []

        # Leg motor pools by segment (T1 front, T2 middle, T3 hind)
        self.mn_leg_fl_L = []
        self.mn_leg_fl_R = []
        self.mn_leg_ml_L = []
        self.mn_leg_ml_R = []
        self.mn_leg_hl_L = []
        self.mn_leg_hl_R = []

        # Proprioceptive & tactile sensory pools
        self.proprio_L_idx = []
        self.proprio_R_idx = []
        self.tactile_L_idx = []
        self.tactile_R_idx = []
        self.descending_idx = []

        # Legacy pool names (backward compatibility with existing code)
        self.mn_b1_L = []
        self.mn_b1_R = []
        self.mn_b2_L = []
        self.mn_b2_R = []
        self.mn_i_L = []
        self.mn_i_R = []
        self.mn_dlm = []
        self.mn_dvm = []

        # Premotor interneuron pools (for monitoring E/I balance)
        self.premotor_exc_idx = []
        self.premotor_inh_idx = []

        # Leg motor pools for walking CPG (by thoracic segment)
        self.mn_leg_L = {1: [], 2: [], 3: []}
        self.mn_leg_R = {1: [], 2: [], 3: []}

        for idx, row in enumerate(self.nodes_df.itertuples()):
            bid = int(row.body_id)
            role = str(getattr(row, 'circuit_role', ''))
            ntype = str(getattr(row, 'type', ''))
            subc = str(getattr(row, 'subclass', ''))
            nt = str(getattr(row, 'consensus_nt', '')).lower()
            side = self._resolve_side(
                getattr(row, 'side', ''),
                getattr(row, 'somaSide', ''),
                getattr(row, 'rootSide', ''),
                getattr(row, 'instance', '')
            )
            is_left = (side == 'L')

            # --- Sensory pools ---
            if role == 'haltere_sensory' or (role == 'mechanosensory' and 'haltere' in subc.lower()):
                (self.haltere_L_idx if is_left else self.haltere_R_idx).append(idx)
            elif role in ('visual_sensory', 'visual_lptc') or 'LPTC' in ntype:
                (self.visual_L_idx if is_left else self.visual_R_idx).append(idx)
            elif role == 'wing_sensory':
                (self.wing_L_idx if is_left else self.wing_R_idx).append(idx)
            elif role == 'proprioceptive_sensory' or 'chordotonal' in subc.lower():
                (self.proprio_L_idx if is_left else self.proprio_R_idx).append(idx)
            elif role == 'tactile_sensory' or 'bristle' in subc.lower() or 'hair plate' in subc.lower():
                (self.tactile_L_idx if is_left else self.tactile_R_idx).append(idx)

            # --- Descending & Premotor interneurons ---
            if role == 'descending_neuron':
                self.descending_idx.append(idx)
            elif role == 'premotor_interneuron':
                if 'gaba' in nt or 'glutamate' in nt:
                    self.premotor_inh_idx.append(idx)
                else:
                    self.premotor_exc_idx.append(idx)

            # --- Motor neuron categorization ---
            nt_lower = ntype.lower()
            subc_lower = subc.lower()

            # Escape jump reflex: Tergotrochanteral muscle motor neurons
            if 'ttm' in nt_lower or 'tergotr' in nt_lower:
                (self.mn_ttm_L if is_left else self.mn_ttm_R).append(idx)

            # Neck / Gaze motor neurons
            if subc_lower == 'nm' or 'neck' in subc_lower or 'cervical' in nt_lower:
                (self.mn_neck_L if is_left else self.mn_neck_R).append(idx)

            # Haltere motor neurons
            if subc_lower == 'hm':
                (self.mn_hm_L if is_left else self.mn_hm_R).append(idx)

            # Leg motor neurons by subclass (fl=front, ml=middle, hl=hind)
            if subc_lower == 'fl':
                (self.mn_leg_fl_L if is_left else self.mn_leg_fl_R).append(idx)
                (self.mn_leg_L[1] if is_left else self.mn_leg_R[1]).append(idx)
            elif subc_lower == 'ml':
                (self.mn_leg_ml_L if is_left else self.mn_leg_ml_R).append(idx)
                (self.mn_leg_L[2] if is_left else self.mn_leg_R[2]).append(idx)
            elif subc_lower == 'hl':
                (self.mn_leg_hl_L if is_left else self.mn_leg_hl_R).append(idx)
                (self.mn_leg_L[3] if is_left else self.mn_leg_R[3]).append(idx)

            # Flight Power muscles: DLM (dorsal longitudinal) — bilateral split
            if 'hdvm' in nt_lower:
                self.mn_hdvm.append(idx)
            elif 'dlm' in nt_lower:
                (self.mn_dlm_L if is_left else self.mn_dlm_R).append(idx)
                self.mn_dlm.append(idx)
            elif 'dvm' in nt_lower:
                (self.mn_dvm_L if is_left else self.mn_dvm_R).append(idx)
                self.mn_dvm.append(idx)

            # Roll actuators: b2, tp1, tp2, tpn (stroke plane angle)
            elif 'b2 mn' in nt_lower:
                (self.mn_roll_L if is_left else self.mn_roll_R).append(idx)
                (self.mn_b2_L if is_left else self.mn_b2_R).append(idx)
            elif any(t in nt_lower for t in ['tp1 mn', 'tp2 mn', 'tpn mn']):
                (self.mn_roll_L if is_left else self.mn_roll_R).append(idx)

            # Pitch actuators: b1, hg1-4, hiii2, iii1 (angle of attack)
            elif 'b1 mn' in nt_lower:
                (self.mn_pitch_L if is_left else self.mn_pitch_R).append(idx)
                (self.mn_b1_L if is_left else self.mn_b1_R).append(idx)
            elif any(t in nt_lower for t in ['hg1', 'hg2', 'hg3', 'hg4', 'hiii2', 'iii1']):
                (self.mn_pitch_L if is_left else self.mn_pitch_R).append(idx)

            # Yaw actuators: i1, i2, ps1, ps2, hi1, hi2 (stroke amplitude)
            elif any(t in nt_lower for t in ['i1 mn', 'i2 mn', 'ps1 mn', 'ps2 mn', 'hi1 mn', 'hi2 mn']):
                (self.mn_yaw_L if is_left else self.mn_yaw_R).append(idx)
                if 'i1 mn' in nt_lower or 'i2 mn' in nt_lower:
                    (self.mn_i_L if is_left else self.mn_i_R).append(idx)

            # Abdominal
            elif 'abd' in nt_lower or 'abdominal' in nt_lower or subc_lower == 'ad':
                if is_left:
                    self.mn_abd_L.append(idx)
                elif side == 'R':
                    self.mn_abd_R.append(idx)
                else:
                    self.mn_abd_pitch.append(idx)

        # Log pool sizes for diagnostics
        self._pool_sizes = {
            'roll_L': len(self.mn_roll_L), 'roll_R': len(self.mn_roll_R),
            'pitch_L': len(self.mn_pitch_L), 'pitch_R': len(self.mn_pitch_R),
            'yaw_L': len(self.mn_yaw_L), 'yaw_R': len(self.mn_yaw_R),
            'DLM_L': len(self.mn_dlm_L), 'DLM_R': len(self.mn_dlm_R),
            'DVM_L': len(self.mn_dvm_L), 'DVM_R': len(self.mn_dvm_R),
            'haltere_L': len(self.haltere_L_idx), 'haltere_R': len(self.haltere_R_idx),
            'premotor_exc': len(self.premotor_exc_idx),
            'premotor_inh': len(self.premotor_inh_idx),
        }

    def _build_synaptic_matrices(self):
        """Construct weighted adjacency matrices for excitation and inhibition."""
        N = self.num_neurons
        # Biological synaptic conductances scaled for circuit density.
        # 148-circuit uses 0.045; 897-circuit uses 0.005; 10k comprehensive mobility circuit uses 0.0035
        if N <= 200:
            g_scale = 0.045
        elif N <= 1500:
            g_scale = 0.005
        else:
            g_scale = 0.0035

        # Precompute ablation sets for fast lookup
        haltere_set = set(self.haltere_L_idx) | set(self.haltere_R_idx) if self.ablate_halteres else set()
        visual_set = set(self.visual_L_idx) | set(self.visual_R_idx) if self.ablate_visual else set()

        if self.use_sparse:
            pre_ids = self.edges_df['pre_id'].values
            post_ids = self.edges_df['post_id'].values
            weights = self.edges_df['weight'].values.astype(np.float32)

            pre_idx = np.array([self.id_to_idx.get(x, -1) for x in pre_ids], dtype=np.int32)
            post_idx = np.array([self.id_to_idx.get(x, -1) for x in post_ids], dtype=np.int32)
            valid = (pre_idx >= 0) & (post_idx >= 0)

            if self.ablated_nodes:
                abl_set = self.ablated_nodes
                valid &= np.array([pre not in abl_set and post not in abl_set for pre, post in zip(pre_ids, post_ids)])

            if haltere_set or visual_set:
                blocked = haltere_set | visual_set
                valid &= np.array([i not in blocked for i in pre_idx])

            pre_idx = pre_idx[valid]
            post_idx = post_idx[valid]
            weights = weights[valid] * g_scale

            nt_arr = np.array([str(self.nodes_df.iloc[idx].get('consensus_nt', '')).lower() for idx in range(N)])
            pre_nts = nt_arr[pre_idx]
            is_inh = (np.char.find(pre_nts, 'gaba') >= 0) | (np.char.find(pre_nts, 'glutamate') >= 0)

            self.W_exc = sp.csr_matrix((weights[~is_inh], (post_idx[~is_inh], pre_idx[~is_inh])), shape=(N, N))
            self.W_inh = sp.csr_matrix((weights[is_inh], (post_idx[is_inh], pre_idx[is_inh])), shape=(N, N))
        else:
            self.W_exc = np.zeros((N, N), dtype=np.float32)
            self.W_inh = np.zeros((N, N), dtype=np.float32)

            for _, edge in self.edges_df.iterrows():
                pre = int(edge['pre_id'])
                post = int(edge['post_id'])
                if pre not in self.id_to_idx or post not in self.id_to_idx:
                    continue
                i = self.id_to_idx[pre]
                j = self.id_to_idx[post]

                if pre in self.ablated_nodes or post in self.ablated_nodes:
                    continue
                if i in haltere_set or i in visual_set:
                    continue

                w = float(edge['weight']) * g_scale
                nt = str(self.nodes_df.iloc[i].get('consensus_nt', '')).lower()

                if 'gaba' in nt or 'glutamate' in nt:
                    self.W_inh[j, i] += w
                else:  # acetylcholine or unknown defaults to excitatory
                    self.W_exc[j, i] += w

    def reset(self):
        """Reset all membrane potentials, conductances, and spike monitors."""
        N = self.num_neurons
        self.v = np.full(N, -65.0, dtype=np.float32)      # Resting potential EL = -65 mV
        self.ge = np.zeros(N, dtype=np.float32)            # Excitatory conductance
        self.gi = np.zeros(N, dtype=np.float32)            # Inhibitory conductance
        self.ref_timer = np.zeros(N, dtype=np.float32)     # Refractory period counter
        self.I_ext = np.zeros(N, dtype=np.float32)         # External input current (mV)

        # Rolling spike buffer for rate estimation (window: 5 ms = 50 steps at dt=0.1ms)
        # Matched to single wing stroke period at 200 Hz flapping to avoid phase lag
        self.spike_window = 50
        self.spike_buffer = np.zeros((self.spike_window, N), dtype=np.bool_)
        self.buffer_ptr = 0
        self.total_spikes = 0

        # Exponential moving average rate estimator
        # tau_rate = 10 ms provides good balance for population pools of 5-10+ neurons
        self.ema_rate = np.full(N, 120.0, dtype=np.float32)
        self.tau_rate = 10.0  # EMA time constant in ms

    def step(self, sensory_inputs: Dict[str, Any]) -> Dict[str, float]:
        """
        Advance spiking network by one control timestep (e.g. 1.0 ms)
        using numerical conductance-based LIF integration.

        Args:
            sensory_inputs: Dict from BiophysicalSensorEncoder with currents/rates.

        Returns:
            Dict of instantaneous motor neuron pool firing rates in Hz.
        """
        N = self.num_neurons

        # 1. Update sensory input currents
        # Baseline calibrated for ~120 Hz resting rate with E/I balance
        # from 300 inhibitory premotor interneurons
        self.I_ext.fill(14.5)

        if not self.ablate_halteres:
            I_h_L = float(sensory_inputs.get('I_haltere_L', 0.45)) * 40.0
            I_h_R = float(sensory_inputs.get('I_haltere_R', 0.45)) * 40.0
            if self.haltere_L_idx:
                self.I_ext[self.haltere_L_idx] += I_h_L
            if self.haltere_R_idx:
                self.I_ext[self.haltere_R_idx] += I_h_R

        if not self.ablate_visual:
            I_v_L = float(sensory_inputs.get('I_vis_L', 0.35)) * 30.0
            I_v_R = float(sensory_inputs.get('I_vis_R', 0.35)) * 30.0
            if self.visual_L_idx:
                self.I_ext[self.visual_L_idx] += I_v_L
            if self.visual_R_idx:
                self.I_ext[self.visual_R_idx] += I_v_R

        I_w_L = float(sensory_inputs.get('I_wing_L', 0.30)) * 25.0
        I_w_R = float(sensory_inputs.get('I_wing_R', 0.30)) * 25.0
        if self.wing_L_idx:
            self.I_ext[self.wing_L_idx] += I_w_L
        if self.wing_R_idx:
            self.I_ext[self.wing_R_idx] += I_w_R

        # Leg ground contact tactile sensory inputs (bristles and hair plates)
        I_touch_L = float(sensory_inputs.get('I_leg_touch_L', 0.0)) * 30.0
        I_touch_R = float(sensory_inputs.get('I_leg_touch_R', 0.0)) * 30.0
        if self.tactile_L_idx:
            self.I_ext[self.tactile_L_idx] += I_touch_L
        if self.tactile_R_idx:
            self.I_ext[self.tactile_R_idx] += I_touch_R

        # Leg joint proprioceptive feedback (FeCO chordotonal organs)
        I_prop_L = float(sensory_inputs.get('I_leg_proprio_L', 0.0)) * 25.0
        I_prop_R = float(sensory_inputs.get('I_leg_proprio_R', 0.0)) * 25.0
        if self.proprio_L_idx:
            self.I_ext[self.proprio_L_idx] += I_prop_L
        if self.proprio_R_idx:
            self.I_ext[self.proprio_R_idx] += I_prop_R

        # Visual looming / predator escape jump trigger via descending neurons
        I_loom = float(sensory_inputs.get('I_visual_loom', 0.0)) * 45.0
        if self.descending_idx and I_loom > 0:
            self.I_ext[self.descending_idx] += I_loom

        # Biophysical parameters
        EL = -65.0      # Resting leak potential (mV)
        Vth = -50.0     # Action potential threshold (mV)
        Vreset = -65.0  # Reset potential (mV)
        Ee = 0.0        # Excitatory reversal potential (mV)
        Ei = -75.0      # Inhibitory reversal potential (mV)
        tau_m = 10.0    # Membrane time constant (ms)
        tau_e = 2.0     # Excitatory conductance decay (ms)
        tau_i = 4.0     # Inhibitory conductance decay (ms)
        t_ref = 1.0     # Refractory period (ms)
        dt_ms = self.dt * 1000.0

        decay_e = np.exp(-dt_ms / tau_e)
        decay_i = np.exp(-dt_ms / tau_i)

        # 2. Integrate neural sub-steps (e.g. 10 sub-steps of 0.1 ms each)
        for _ in range(self.steps_per_call):
            # Conductance decay
            self.ge *= decay_e
            self.gi *= decay_i

            # Refractory timer decrement
            self.ref_timer = np.maximum(0.0, self.ref_timer - dt_ms)
            active_mask = (self.ref_timer <= 0.0)

            # Membrane potential integration: dv/dt = (-(v - EL) - ge*(v - Ee) - gi*(v - Ei) + I_ext) / tau_m
            dv = (-(self.v - EL) - self.ge * (self.v - Ee) - self.gi * (self.v - Ei) + self.I_ext) * (dt_ms / tau_m)
            self.v[active_mask] += dv[active_mask]

            # Spike detection
            spiked = (self.v >= Vth) & active_mask
            if np.any(spiked):
                self.v[spiked] = Vreset
                self.ref_timer[spiked] = t_ref

                # Synaptic transmission: ge += W_exc * spikes, gi += W_inh * spikes
                spike_vec = spiked.astype(np.float32)
                self.ge += self.W_exc @ spike_vec
                self.gi += self.W_inh @ spike_vec

            # Store in rolling buffer
            self.spike_buffer[self.buffer_ptr, :] = spiked
            self.buffer_ptr = (self.buffer_ptr + 1) % self.spike_window
            self.total_spikes += int(np.sum(spiked))

        # 3. Compute smoothed motor neuron firing rates (Hz) using EMA
        dt_ms_step = self.step_dt * 1000.0
        alpha_rate = dt_ms_step / (self.tau_rate + dt_ms_step)
        T_window = self.spike_window * self.dt
        total_spikes_in_window = np.sum(self.spike_buffer, axis=0)
        window_rates = total_spikes_in_window / T_window
        self.ema_rate += alpha_rate * (window_rates - self.ema_rate)
        rates = self.ema_rate

        def pool_rate(indices, default=120.0):
            return float(np.mean(rates[indices])) if indices else default

        return {
            # Grouped functional actuator pools (5-10+ neurons each for stable estimates)
            'roll_L': pool_rate(self.mn_roll_L),
            'roll_R': pool_rate(self.mn_roll_R),
            'pitch_L': pool_rate(self.mn_pitch_L),
            'pitch_R': pool_rate(self.mn_pitch_R),
            'yaw_L': pool_rate(self.mn_yaw_L),
            'yaw_R': pool_rate(self.mn_yaw_R),
            # Bilateral power pools
            'DLM_L': pool_rate(self.mn_dlm_L, 400.0),
            'DLM_R': pool_rate(self.mn_dlm_R, 400.0),
            'DVM_L': pool_rate(self.mn_dvm_L, 400.0),
            'DVM_R': pool_rate(self.mn_dvm_R, 400.0),
            # Legacy pool names (backward compatibility)
            'b1_L': pool_rate(self.mn_b1_L),
            'b1_R': pool_rate(self.mn_b1_R),
            'b2_L': pool_rate(self.mn_b2_L),
            'b2_R': pool_rate(self.mn_b2_R),
            'i1_i2_L': pool_rate(self.mn_i_L),
            'i1_i2_R': pool_rate(self.mn_i_R),
            'DLM': pool_rate(self.mn_dlm, 400.0),
            'DVM': pool_rate(self.mn_dvm, 400.0),
            # Abdominal
            'abd_L': pool_rate(self.mn_abd_L),
            'abd_R': pool_rate(self.mn_abd_R),
            'abd_pitch': pool_rate(self.mn_abd_pitch),
            # Leg motor pools by thoracic segment (fl=front T1, ml=middle T2, hl=hind T3)
            'leg_fl_L': pool_rate(self.mn_leg_fl_L, 30.0),
            'leg_fl_R': pool_rate(self.mn_leg_fl_R, 30.0),
            'leg_ml_L': pool_rate(self.mn_leg_ml_L, 30.0),
            'leg_ml_R': pool_rate(self.mn_leg_ml_R, 30.0),
            'leg_hl_L': pool_rate(self.mn_leg_hl_L, 30.0),
            'leg_hl_R': pool_rate(self.mn_leg_hl_R, 30.0),
            # Escape jump reflex (TTMn)
            'ttm_jump_L': pool_rate(self.mn_ttm_L, 30.0),
            'ttm_jump_R': pool_rate(self.mn_ttm_R, 30.0),
            # Head gaze / neck
            'neck_L': pool_rate(self.mn_neck_L, 100.0),
            'neck_R': pool_rate(self.mn_neck_R, 100.0),
            # Haltere active motor tuning
            'hm_L': pool_rate(self.mn_hm_L, 100.0),
            'hm_R': pool_rate(self.mn_hm_R, 100.0),
            # Diagnostics
            'premotor_exc': pool_rate(self.premotor_exc_idx),
            'premotor_inh': pool_rate(self.premotor_inh_idx),
        }
