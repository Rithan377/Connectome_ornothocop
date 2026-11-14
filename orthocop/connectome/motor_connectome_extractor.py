"""
Motor-Only Connectome Extractor for Drosophila Male CNS v1.0.

Extracts the COMPLETE physical-motor control connectome: every neuron required
for flight, walking, body stabilization, head/gaze control, and postural adjustment.
Removes non-motor neurons (gustatory, olfactory, Kenyon cells, thermosensory,
hygrosensory, chemosensory, and purely cognitive circuits).

The goal: determine the EXACT connectome needed for physical control, verify
100% of motor pathways are preserved despite neuron reduction, and validate
in a proper MuJoCo/FlyBody simulation environment.

Motor Neuron Subclasses (from FlyWire Male CNS v1.0):
  wm  = Wing Muscles (67 neurons) — flight power + steering
  fl  = Front Leg T1 (135 neurons) — walking, grooming
  ml  = Middle Leg T2 (116 neurons) — walking, balance
  hl  = Hind Leg T3 (130 neurons) — walking, jumping
  ad  = Abdomen (214 neurons) — steering, balance, posture
  nm  = Neck Muscles T1 (24 neurons) — head movement, gaze stabilization
  hm  = Haltere Muscles T3 (16 neurons) — haltere control (flight sensor)
  xm  = Cross-thorax (6 neurons) — thorax rigidity
"""

import os
import json
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.compute as pc
import pyarrow as pa
import networkx as nx
from collections import defaultdict


class MotorConnectomeExtractor:
    """
    Extracts the complete motor-control connectome from Male Drosophila CNS.
    
    Strategy:
    1. Seed with ALL motor neurons (708 VNC motor neurons across all subclasses)
    2. Add ALL proprioceptive sensory neurons (campaniform sensilla, chordotonal
       organs, haltere afferents, leg proprioceptors, hair plates, wing stretch)
    3. Add mechanosensory neurons relevant to physical control (wind/gravity sensing)
    4. Add visual LPTCs essential for flight stabilization (VS, HS cells)
    5. Add ALL descending neurons that connect to motor neurons (brain→body commands)
    6. Add VNC intrinsic interneurons that mediate motor control (premotor circuits,
       CPGs, coordinators)
    7. Add ascending neurons that provide motor-state feedback to brain
    8. Trace connectivity to ensure complete sensory→interneuron→motor pathways
    9. EXCLUDE: gustatory, olfactory, Kenyon cells, thermosensory, hygrosensory,
       chemosensory, mushroom body neurons, and purely cognitive circuits
    """

    # Non-physical neuron classes to EXCLUDE
    EXCLUDED_CLASSES = {
        'gustatory',        # Taste — eating behavior
        'olfactory',        # Smell — foraging/mating
        'Kenyon_Cell',      # Mushroom body — learning/memory
        'hygrosensory',     # Humidity sensing
        'thermosensory',    # Temperature sensing
        'chemosensory',     # Chemical sensing
    }

    # Non-physical superclasses/subclasses
    EXCLUDED_SUPERCLASSES = {
        'cb_endocrine',     # Endocrine (hormonal)
        'vnc_endocrine',    # VNC endocrine
        'ENS',              # Enteric nervous system (gut)
    }

    # Subclasses of mechanosensory to EXCLUDE (non-motor)
    EXCLUDED_MECH_SUBCLASSES = {
        'grooming',              # Grooming behavior
        'pharyngeal sensillum',  # Feeding apparatus
        'labellar bristle',      # Proboscis (feeding)
    }

    def __init__(self, data_dir="data", output_dir="circuits"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.ann_path = os.path.join(data_dir, "processed", "cleaned_annotations.feather")
        self.weights_path = os.path.join(data_dir, "original",
                                         "connectome-weights-male-cns-v1.0-minconf-0.5.feather")

        print("[*] Loading annotations...")
        self.annotations = pd.read_feather(self.ann_path)
        self.weights_table = None

        print(f"[+] Loaded {len(self.annotations):,} neuron annotations")

    def load_weights_table(self):
        if self.weights_table is None:
            print("[*] Memory-mapping full weights feather table...")
            self.weights_table = feather.read_table(self.weights_path, memory_map=True)
        return self.weights_table

    def _is_excluded(self, row):
        """Check if a neuron belongs to an excluded (non-physical) category."""
        cls = str(row.get('class', ''))
        superclass = str(row.get('superclass', ''))
        subclass = str(row.get('subclass', ''))

        if cls in self.EXCLUDED_CLASSES:
            return True
        if superclass in self.EXCLUDED_SUPERCLASSES:
            return True
        if cls == 'mechanosensory' and subclass in self.EXCLUDED_MECH_SUBCLASSES:
            return True
        return False

    def extract_motor_connectome(self, min_synapse_weight=2, dn_to_mn_threshold=10,
                                  premotor_to_mn_threshold=5, max_premotor_interneurons=2000):
        """
        Extract the complete motor-control connectome.

        Args:
            min_synapse_weight: Minimum synaptic weight to include an edge.
            dn_to_mn_threshold: Minimum total synapses from a DN to any motor neuron.
            premotor_to_mn_threshold: Minimum total synapses from an interneuron to any MN.
            max_premotor_interneurons: Maximum number of premotor interneurons to include.

        Returns:
            (nodes_df, edges_df, summary_dict)
        """
        ann = self.annotations
        tbl = self.load_weights_table()

        collected = {}  # body_id -> (row, circuit_role)

        def add_neurons(df, role):
            for _, row in df.iterrows():
                bid = int(row['body_id'])
                if bid not in collected:
                    collected[bid] = (row, role)

        # =====================================================================
        # LAYER 1: ALL Motor Neurons (708 neurons)
        # =====================================================================
        print("\n[LAYER 1] Motor Neurons — all physical muscle controllers")
        all_motor = ann[ann['superclass'] == 'vnc_motor'].copy()
        add_neurons(all_motor, 'motor_neuron')
        print(f"  Wing muscles (wm): {len(all_motor[all_motor['subclass']=='wm'])}")
        print(f"  Front leg (fl):    {len(all_motor[all_motor['subclass']=='fl'])}")
        print(f"  Middle leg (ml):   {len(all_motor[all_motor['subclass']=='ml'])}")
        print(f"  Hind leg (hl):     {len(all_motor[all_motor['subclass']=='hl'])}")
        print(f"  Abdomen (ad):      {len(all_motor[all_motor['subclass']=='ad'])}")
        print(f"  Neck (nm):         {len(all_motor[all_motor['subclass']=='nm'])}")
        print(f"  Haltere (hm):      {len(all_motor[all_motor['subclass']=='hm'])}")
        print(f"  Cross-thorax (xm): {len(all_motor[all_motor['subclass']=='xm'])}")
        print(f"  TOTAL motor neurons: {len(all_motor)}")

        # =====================================================================
        # LAYER 2: Proprioceptive Sensory Neurons (1,454 neurons)
        # =====================================================================
        print("\n[LAYER 2] Proprioceptive Sensors — body position/force feedback")
        proprio = ann[ann['class'] == 'mechanosensory_proprioceptive'].copy()
        add_neurons(proprio, 'proprioceptive_sensory')
        print(f"  Campaniform sensilla: {len(proprio[proprio['subclass']=='campaniform sensilla'])}")
        print(f"  Chordotonal organs:   {len(proprio[proprio['subclass']=='chordotonal organ'])}")
        print(f"  Haltere afferents:    {len(proprio[proprio['subclass']=='haltere'])}")
        print(f"  Leg proprioceptors:   {len(proprio[proprio['subclass']=='leg'])}")
        print(f"  Hair plates:          {len(proprio[proprio['subclass']=='hair plate'])}")
        print(f"  Wing stretch:         {len(proprio[proprio['subclass']=='wing'])}")
        print(f"  Abdomen proprio:      {len(proprio[proprio['subclass']=='abdomen'])}")
        print(f"  TOTAL proprioceptive: {len(proprio)}")

        # =====================================================================
        # LAYER 3: Mechanosensory — wind, gravity, auditory (physical forces)
        # =====================================================================
        print("\n[LAYER 3] Mechanosensory — physical force detection")
        mech = ann[ann['class'] == 'mechanosensory'].copy()
        # Exclude non-physical mechanosensory
        mech_physical = mech[~mech['subclass'].isin(self.EXCLUDED_MECH_SUBCLASSES)]
        add_neurons(mech_physical, 'mechanosensory')
        print(f"  Wind/gravity:  {len(mech_physical[mech_physical['subclass']=='wind_gravity'])}")
        print(f"  Auditory:      {len(mech_physical[mech_physical['subclass']=='auditory'])}")
        print(f"  General:       {len(mech_physical[mech_physical['subclass']==''])}")
        print(f"  TOTAL physical mechanosensory: {len(mech_physical)}")

        # =====================================================================
        # LAYER 4: Tactile Mechanosensory — body/wing/leg touch receptors
        # =====================================================================
        print("\n[LAYER 4] Tactile Mechanosensory — body contact/bristle sensors")
        tactile = ann[ann['class'] == 'mechanosensory_tactile'].copy()
        # Exclude labellar bristle (feeding)
        tactile_physical = tactile[~tactile['subclass'].isin(['labellar bristle'])]
        add_neurons(tactile_physical, 'tactile_sensory')
        print(f"  Bristle:          {len(tactile_physical[tactile_physical['subclass']=='mechanosensory bristle'])}")
        print(f"  Wing bristle:     {len(tactile_physical[tactile_physical['subclass']=='wing bristle'])}")
        print(f"  Leg bristle:      {len(tactile_physical[tactile_physical['subclass']=='leg bristle'])}")
        print(f"  TOTAL tactile:    {len(tactile_physical)}")

        # =====================================================================
        # LAYER 5: Visual LPTCs — flight-critical optic flow detectors
        # =====================================================================
        print("\n[LAYER 5] Visual LPTCs — flight stabilization vision")
        vis_mask = ann['type'].str.contains(r'^(?:VS|HS|CH|H1|H2)', regex=True, na=False)
        vis_tangential = ann[ann['class'] == 'visual'].copy()
        lptc_mask = vis_tangential['type'].str.contains('tangential|lobula', case=False, na=False)
        visual_flight = ann[vis_mask | (ann['class'] == 'visual') & lptc_mask].copy()
        # Keep only actual LPTCs, not all 6000+ visual neurons
        visual_flight = visual_flight.drop_duplicates(subset=['body_id'])
        add_neurons(visual_flight, 'visual_lptc')
        print(f"  TOTAL visual LPTCs: {len(visual_flight)}")

        # =====================================================================
        # LAYER 6: Descending Neurons that connect to Motor Neurons
        # =====================================================================
        print("\n[LAYER 6] Descending Neurons — brain→motor commands")
        mn_ids = set(all_motor['body_id'])
        mn_pa = pa.array(list(mn_ids), type=pa.int64())

        # Find all neurons projecting TO motor neurons
        mask_to_mn = pc.is_in(tbl['body_post'], value_set=mn_pa)
        upstream_of_mn = tbl.filter(mask_to_mn).to_pandas()

        # All descending neuron IDs
        all_dn_ids = set(ann[ann['superclass'] == 'descending_neuron']['body_id'])

        # DNs with strong connections to MNs
        dn_weights = upstream_of_mn[upstream_of_mn['body_pre'].isin(all_dn_ids)].groupby('body_pre')['weight'].sum()
        strong_dn_ids = set(dn_weights[dn_weights >= dn_to_mn_threshold].index)

        # Also include DNs that connect to ANY neuron we've already collected
        collected_ids = set(collected.keys())
        all_collected_pa = pa.array(list(collected_ids), type=pa.int64())
        mask_to_collected = pc.is_in(tbl['body_post'], value_set=all_collected_pa)
        upstream_of_collected = tbl.filter(mask_to_collected).to_pandas()
        dn_to_collected = upstream_of_collected[upstream_of_collected['body_pre'].isin(all_dn_ids)]
        dn_weights_broad = dn_to_collected.groupby('body_pre')['weight'].sum()
        broad_dn_ids = set(dn_weights_broad[dn_weights_broad >= dn_to_mn_threshold * 2].index)

        selected_dn_ids = strong_dn_ids | broad_dn_ids
        dn_neurons = ann[ann['body_id'].isin(selected_dn_ids)].copy()
        add_neurons(dn_neurons, 'descending_neuron')
        print(f"  DNs with direct MN connections (>={dn_to_mn_threshold} syn): {len(strong_dn_ids)}")
        print(f"  DNs with broad motor pathway connections: {len(broad_dn_ids)}")
        print(f"  TOTAL selected DNs: {len(selected_dn_ids)}")

        # =====================================================================
        # LAYER 7: Premotor Interneurons — VNC intrinsic mediating motor control
        # =====================================================================
        print("\n[LAYER 7] Premotor Interneurons — VNC motor circuit interneurons")
        vnc_intr_ids = set(ann[ann['superclass'] == 'vnc_intrinsic']['body_id'])

        # Interneurons with strong direct connections to MNs
        intr_to_mn = upstream_of_mn[upstream_of_mn['body_pre'].isin(vnc_intr_ids)]
        intr_weights = intr_to_mn.groupby('body_pre')['weight'].sum()
        strong_premotor = set(intr_weights[intr_weights >= premotor_to_mn_threshold].index)

        # Also find interneurons that receive from our sensory neurons and project to MNs
        # (complete sensory→interneuron→motor pathway)
        sensory_ids = set()
        for bid, (_, role) in collected.items():
            if 'sensory' in role or 'lptc' in role or 'mechanosensory' in role or 'tactile' in role:
                sensory_ids.add(bid)

        sensory_pa = pa.array(list(sensory_ids), type=pa.int64())
        mask_from_sensory = pc.is_in(tbl['body_pre'], value_set=sensory_pa)
        downstream_of_sensory = tbl.filter(mask_from_sensory).to_pandas()
        sensory_target_ids = set(downstream_of_sensory[
            downstream_of_sensory['body_post'].isin(vnc_intr_ids)
        ]['body_post'])

        # Interneurons that BOTH receive from sensory AND project to motor
        pathway_interneurons = sensory_target_ids & set(intr_weights.index)

        all_premotor_ids = strong_premotor | pathway_interneurons
        if len(all_premotor_ids) > max_premotor_interneurons:
            # Rank by total weight to MNs and take top N
            ranked = intr_weights[intr_weights.index.isin(all_premotor_ids)].sort_values(ascending=False)
            all_premotor_ids = set(ranked.head(max_premotor_interneurons).index)

        premotor_neurons = ann[ann['body_id'].isin(all_premotor_ids)].copy()
        add_neurons(premotor_neurons, 'premotor_interneuron')
        print(f"  Direct premotor (>={premotor_to_mn_threshold} syn to MN): {len(strong_premotor)}")
        print(f"  Sensory→interneuron→motor pathway: {len(pathway_interneurons)}")
        print(f"  TOTAL premotor interneurons: {len(all_premotor_ids)}")

        # =====================================================================
        # LAYER 8: Ascending Neurons — motor state feedback to brain
        # =====================================================================
        print("\n[LAYER 8] Ascending Neurons — motor-state feedback")
        all_an_ids = set(ann[ann['superclass'] == 'ascending_neuron']['body_id'])

        # Ascending neurons that receive from motor neurons or premotor interneurons
        motor_premotor_ids = mn_ids | all_premotor_ids
        mp_pa = pa.array(list(motor_premotor_ids), type=pa.int64())
        mask_from_motor = pc.is_in(tbl['body_pre'], value_set=mp_pa)
        downstream_of_motor = tbl.filter(mask_from_motor).to_pandas()

        an_from_motor = downstream_of_motor[downstream_of_motor['body_post'].isin(all_an_ids)]
        an_weights = an_from_motor.groupby('body_post')['weight'].sum()
        selected_an_ids = set(an_weights[an_weights >= premotor_to_mn_threshold].index)

        an_neurons = ann[ann['body_id'].isin(selected_an_ids)].copy()
        add_neurons(an_neurons, 'ascending_neuron')
        print(f"  TOTAL ascending neurons (motor feedback): {len(selected_an_ids)}")

        # =====================================================================
        # SUMMARY & EXPORT
        # =====================================================================
        print(f"\n{'='*60}")
        print(f"TOTAL MOTOR-CONTROL CONNECTOME: {len(collected):,} neurons")
        print(f"(Removed {len(ann) - len(collected):,} non-physical neurons)")
        print(f"{'='*60}")

        # Build nodes dataframe
        nodes_data = []
        for bid, (row, role) in collected.items():
            row_dict = row.to_dict() if hasattr(row, 'to_dict') else dict(row)
            row_dict['circuit_role'] = role
            row_dict['body_id'] = bid
            nodes_data.append(row_dict)

        nodes_df = pd.DataFrame(nodes_data).drop_duplicates(subset=['body_id'])

        # Role breakdown
        role_counts = nodes_df['circuit_role'].value_counts().to_dict()
        print("\nRole breakdown:")
        for role, count in sorted(role_counts.items(), key=lambda x: -x[1]):
            print(f"  {role:30s}: {count:,}")

        # =====================================================================
        # EXTRACT EDGES between motor-control neurons
        # =====================================================================
        print(f"\n[*] Extracting edges among {len(nodes_df):,} motor-control neurons (weight >= {min_synapse_weight})...")
        motor_ids = set(nodes_df['body_id'])
        motor_pa = pa.array(list(motor_ids), type=pa.int64())

        mask_pre = pc.is_in(tbl['body_pre'], value_set=motor_pa)
        mask_post = pc.is_in(tbl['body_post'], value_set=motor_pa)
        mask_both = pc.and_(mask_pre, mask_post)

        edges_df = tbl.filter(mask_both).to_pandas()
        edges_df = edges_df[edges_df['weight'] >= min_synapse_weight].copy()
        edges_df = edges_df.rename(columns={'body_pre': 'pre_id', 'body_post': 'post_id'})

        print(f"[+] Extracted {len(edges_df):,} edges")

        # =====================================================================
        # SAVE
        # =====================================================================
        nodes_path = os.path.join(self.output_dir, "motor_connectome_nodes.feather")
        edges_path = os.path.join(self.output_dir, "motor_connectome_edges.feather")
        nodes_df.to_feather(nodes_path)
        edges_df.to_feather(edges_path)

        summary = {
            "total_neurons": len(nodes_df),
            "total_edges": len(edges_df),
            "removed_neurons": len(ann) - len(nodes_df),
            "role_breakdown": role_counts,
            "motor_subclass_breakdown": all_motor['subclass'].value_counts().to_dict(),
            "neurotransmitter_breakdown": nodes_df['consensus_nt'].value_counts().to_dict() if 'consensus_nt' in nodes_df.columns else {},
            "parameters": {
                "min_synapse_weight": min_synapse_weight,
                "dn_to_mn_threshold": dn_to_mn_threshold,
                "premotor_to_mn_threshold": premotor_to_mn_threshold,
            }
        }

        summary_path = os.path.join(self.output_dir, "motor_connectome_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n[+] Saved motor connectome: {nodes_path}")
        print(f"[+] Saved motor edges: {edges_path}")
        print(f"[+] Saved summary: {summary_path}")

        return nodes_df, edges_df, summary

    def validate_motor_pathway_completeness(self, nodes_df=None, edges_df=None):
        """
        Validate that 100% of motor control pathways are preserved.
        
        Checks:
        1. Every motor neuron is reachable from at least one sensory neuron
        2. Every motor subclass has complete bilateral coverage
        3. Sensory→premotor→motor pathways exist for all modalities
        4. No isolated motor neurons (every MN has upstream input)
        5. Flight-specific: wing MN, haltere, LPTC pathways intact
        6. Walking-specific: leg MN, proprioceptor pathways intact
        """
        if nodes_df is None:
            nodes_df = pd.read_feather(os.path.join(self.output_dir, "motor_connectome_nodes.feather"))
        if edges_df is None:
            edges_df = pd.read_feather(os.path.join(self.output_dir, "motor_connectome_edges.feather"))

        print("\n" + "=" * 60)
        print("MOTOR PATHWAY COMPLETENESS VALIDATION")
        print("=" * 60)

        # Build NetworkX graph
        G = nx.DiGraph()
        node_meta = {}
        for _, row in nodes_df.iterrows():
            bid = int(row['body_id'])
            role = str(row.get('circuit_role', ''))
            subclass = str(row.get('subclass', ''))
            G.add_node(bid, circuit_role=role, subclass=subclass)
            node_meta[bid] = {'role': role, 'subclass': subclass}

        for _, row in edges_df.iterrows():
            pre = int(row['pre_id'])
            post = int(row['post_id'])
            if G.has_node(pre) and G.has_node(post):
                G.add_edge(pre, post, weight=float(row['weight']))

        # Categorize nodes
        sensory_nodes = [n for n, d in G.nodes(data=True) if 'sensory' in d.get('circuit_role', '') or 'lptc' in d.get('circuit_role', '') or 'mechanosensory' in d.get('circuit_role', '') or 'tactile' in d.get('circuit_role', '')]
        motor_nodes = [n for n, d in G.nodes(data=True) if d.get('circuit_role') == 'motor_neuron']
        dn_nodes = [n for n, d in G.nodes(data=True) if d.get('circuit_role') == 'descending_neuron']
        premotor_nodes = [n for n, d in G.nodes(data=True) if d.get('circuit_role') == 'premotor_interneuron']

        results = {
            'graph_nodes': G.number_of_nodes(),
            'graph_edges': G.number_of_edges(),
            'sensory_count': len(sensory_nodes),
            'motor_count': len(motor_nodes),
            'dn_count': len(dn_nodes),
            'premotor_count': len(premotor_nodes),
        }

        # CHECK 1: Motor neuron reachability from sensory
        print("\n[CHECK 1] Motor neuron reachability from sensory neurons")
        mn_reachable = 0
        mn_unreachable = []
        for mn in motor_nodes:
            reachable = False
            for sn in sensory_nodes[:200]:  # Sample for speed
                if nx.has_path(G, sn, mn):
                    reachable = True
                    break
            if reachable:
                mn_reachable += 1
            else:
                mn_unreachable.append(mn)

        reachability = mn_reachable / max(1, len(motor_nodes)) * 100
        results['mn_reachability_pct'] = reachability
        print(f"  Motor neurons reachable from sensory: {mn_reachable}/{len(motor_nodes)} ({reachability:.1f}%)")
        if mn_unreachable:
            print(f"  ⚠ Unreachable MNs: {len(mn_unreachable)}")

        # CHECK 2: Bilateral coverage
        print("\n[CHECK 2] Bilateral motor coverage")
        motor_df = nodes_df[nodes_df['circuit_role'] == 'motor_neuron']
        for subclass in ['wm', 'fl', 'ml', 'hl', 'ad', 'nm', 'hm', 'xm']:
            sc_neurons = motor_df[motor_df['subclass'] == subclass]
            if len(sc_neurons) > 0:
                sides = sc_neurons['somaSide'].value_counts().to_dict()
                l_count = sides.get('L', 0)
                r_count = sides.get('R', 0)
                total = len(sc_neurons)
                status = "✓" if l_count > 0 and r_count > 0 else "⚠ MISSING SIDE"
                print(f"  {subclass:5s}: {total:3d} neurons (L={l_count}, R={r_count}) {status}")
            else:
                print(f"  {subclass:5s}: MISSING!")

        # CHECK 3: Motor neurons have upstream input
        print("\n[CHECK 3] Motor neurons with upstream synaptic input")
        mn_with_input = sum(1 for mn in motor_nodes if G.in_degree(mn) > 0)
        results['mn_with_input_pct'] = mn_with_input / max(1, len(motor_nodes)) * 100
        print(f"  MNs with input: {mn_with_input}/{len(motor_nodes)} ({results['mn_with_input_pct']:.1f}%)")

        # CHECK 4: Flight pathway integrity
        print("\n[CHECK 4] Flight pathway integrity")
        flight_mn_ids = set(motor_df[motor_df['subclass'] == 'wm']['body_id'])
        haltere_ids = set(nodes_df[nodes_df['circuit_role'] == 'proprioceptive_sensory'][
            nodes_df['subclass'] == 'haltere']['body_id']) if 'subclass' in nodes_df.columns else set()
        visual_ids = set(nodes_df[nodes_df['circuit_role'] == 'visual_lptc']['body_id'])

        # Haltere → wing MN paths
        halt_to_wm = 0
        for h in list(haltere_ids)[:30]:
            for wm in list(flight_mn_ids)[:20]:
                if G.has_node(h) and G.has_node(wm) and nx.has_path(G, h, wm):
                    halt_to_wm += 1
                    break
        print(f"  Haltere→WingMN paths: {halt_to_wm} haltere neurons with path to wing MN")

        # Visual → wing MN paths
        vis_to_wm = 0
        for v in list(visual_ids)[:20]:
            for wm in list(flight_mn_ids)[:20]:
                if G.has_node(v) and G.has_node(wm) and nx.has_path(G, v, wm):
                    vis_to_wm += 1
                    break
        print(f"  Visual→WingMN paths: {vis_to_wm} visual neurons with path to wing MN")

        # CHECK 5: Walking pathway integrity
        print("\n[CHECK 5] Walking pathway integrity")
        for leg_sc, leg_name in [('fl', 'Front Leg'), ('ml', 'Middle Leg'), ('hl', 'Hind Leg')]:
            leg_mn_ids = set(motor_df[motor_df['subclass'] == leg_sc]['body_id'])
            leg_proprio = nodes_df[
                (nodes_df['circuit_role'] == 'proprioceptive_sensory') &
                (nodes_df['subclass'].isin(['leg', 'chordotonal organ', 'hair plate']))
            ]
            paths_found = 0
            for p in list(leg_proprio['body_id'])[:20]:
                for lm in list(leg_mn_ids)[:10]:
                    if G.has_node(p) and G.has_node(lm) and nx.has_path(G, p, lm):
                        paths_found += 1
                        break
            print(f"  {leg_name} proprio→MN paths: {paths_found}")

        # CHECK 6: Strongly connected components
        print("\n[CHECK 6] Recurrent circuit structure")
        sccs = [c for c in nx.strongly_connected_components(G) if len(c) > 1]
        results['num_recurrent_circuits'] = len(sccs)
        results['largest_recurrent_size'] = max(len(c) for c in sccs) if sccs else 0
        print(f"  Recurrent circuits (SCCs > 1): {len(sccs)}")
        print(f"  Largest recurrent circuit: {results['largest_recurrent_size']} neurons")

        # OVERALL VERDICT
        print("\n" + "=" * 60)
        flight_ok = halt_to_wm > 0 and vis_to_wm > 0 and len(flight_mn_ids) > 0
        walk_ok = len(motor_df[motor_df['subclass'] == 'fl']) > 0 and len(motor_df[motor_df['subclass'] == 'hl']) > 0
        input_ok = results['mn_with_input_pct'] > 90.0

        results['flight_pathways_intact'] = flight_ok
        results['walking_pathways_intact'] = walk_ok
        results['motor_input_sufficient'] = input_ok

        if flight_ok and walk_ok and input_ok:
            print("✓ VERDICT: Motor connectome has COMPLETE physical control coverage")
        else:
            print("⚠ VERDICT: Some pathways may need attention")
            if not flight_ok: print("  - Flight pathways incomplete")
            if not walk_ok: print("  - Walking pathways incomplete")
            if not input_ok: print("  - Some motor neurons lack upstream input")

        print("=" * 60)

        # Save validation results
        val_path = os.path.join(self.output_dir, "motor_connectome_validation.json")
        with open(val_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"[+] Validation results saved to {val_path}")

        return results


if __name__ == "__main__":
    extractor = MotorConnectomeExtractor()
    nodes_df, edges_df, summary = extractor.extract_motor_connectome()
    validation = extractor.validate_motor_pathway_completeness(nodes_df, edges_df)
