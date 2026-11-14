"""
Build Comprehensive Drosophila Mobility Connectome Circuit.

Combines:
1. Stabilized Flight Circuit (897 neurons):
   - 74 sensory neurons (haltere, visual, wing)
   - 24 descending neurons
   - 62 wing flight motor neurons (all 23 flight muscle types)
   - 737 flight premotor interneurons (247 GABAergic/glutamatergic inhibitory)
2. All 6-Leg Motor & Premotor Systems (T1 Front, T2 Middle, T3 Hind):
   - 381 leg motor neurons (coxa, trochanter, femur, tibia, tarsus)
   - Tergotrochanteral Muscle (TTMn) for high-power escape jump kick
   - Leg Central Pattern Generator (CPG) premotor interneurons (tripod walking)
3. Complete Leg Proprioception & Ground Contact Afferents:
   - 1,458 Chordotonal organ (FeCO) joint position afferents
   - 2,558 Tactile bristles and hair plates for ground contact detection
   - Campaniform sensilla strain gauges
4. Head Gaze & Neck Motor System:
   - 24 neck motor neurons for head stabilization during motion
5. Abdominal Steering & Landing Trim:
   - 214 abdominal motor neurons (MNad01-MNad27) for pitch trim, rudder, and landing
6. Dynamic Haltere Actuation:
   - 16 haltere motor neurons
7. Full Descending Motor Control Pathways:
   - 898 Descending Neurons (DNs) coordinating walking, turning, jumping, and flight

Outputs:
- circuits/comprehensive_mobility_nodes.feather
- circuits/comprehensive_mobility_edges.feather
- circuits/comprehensive_mobility_summary.json
"""

import os
import json
import numpy as np
import pandas as pd
import pyarrow.feather as feather


def build_comprehensive_mobility_circuit():
    print("=" * 70)
    print(" BUILDING COMPREHENSIVE DROSOPHILA FULL-BODY MOBILITY CONNECTOME")
    print("=" * 70)

    # 1. Load source datasets
    stab_nodes = feather.read_table('circuits/stabilized_flight_nodes.feather').to_pandas()
    stab_edges = feather.read_table('circuits/stabilized_flight_edges.feather').to_pandas()
    motor_nodes = feather.read_table('circuits/motor_connectome_nodes.feather').to_pandas()
    motor_edges = feather.read_table('circuits/motor_connectome_edges.feather').to_pandas()

    print(f"[*] Loaded baseline flight circuit: {len(stab_nodes)} nodes, {len(stab_edges)} edges")
    print(f"[*] Loaded motor connectome: {len(motor_nodes)} nodes, {len(motor_edges)} edges")

    # 2. Merge nodes ensuring exact flight roles are preserved
    # Use motor_nodes as base, but ensure any specialized flight annotations from stab_nodes are retained
    stab_dict = {row['body_id']: row for _, row in stab_nodes.iterrows()}
    
    # Identify nodes in stab_nodes missing from motor_nodes
    missing_from_motor = [bid for bid in stab_dict if bid not in set(motor_nodes['body_id'])]
    if missing_from_motor:
        print(f"[*] Adding {len(missing_from_motor)} missing specialized sensory nodes: {missing_from_motor}")
        extra_df = stab_nodes[stab_nodes['body_id'].isin(missing_from_motor)]
        combined_nodes = pd.concat([motor_nodes, extra_df], ignore_index=True)
    else:
        combined_nodes = motor_nodes.copy()

    # Update circuit_role for flight nodes to ensure haltere_sensory, visual_sensory, wing_sensory are exact
    for idx, row in combined_nodes.iterrows():
        bid = row['body_id']
        if bid in stab_dict:
            stab_role = stab_dict[bid].get('circuit_role')
            if stab_role and str(stab_role) != 'nan':
                combined_nodes.at[idx, 'circuit_role'] = stab_role

    print(f"[+] Total merged mobility nodes: {len(combined_nodes)}")

    # 3. Merge edges
    merged_edges = pd.concat([stab_edges, motor_edges], ignore_index=True)
    merged_edges = merged_edges.drop_duplicates(subset=['pre_id', 'post_id'], keep='first')
    
    # Filter edges to only include nodes present in combined_nodes
    valid_bids = set(combined_nodes['body_id'])
    merged_edges = merged_edges[
        merged_edges['pre_id'].isin(valid_bids) & 
        merged_edges['post_id'].isin(valid_bids)
    ].reset_index(drop=True)
    
    print(f"[+] Total merged mobility synapses: {len(merged_edges)}")

    # 4. Generate summary breakdown
    roles = combined_nodes['circuit_role'].value_counts().to_dict()
    subclasses = combined_nodes['subclass'].value_counts().to_dict()
    nts = combined_nodes['consensus_nt'].value_counts().to_dict()

    mn_nodes = combined_nodes[combined_nodes['circuit_role'] == 'motor_neuron']
    mn_breakdown = mn_nodes['subclass'].value_counts().to_dict()

    summary = {
        'total_neurons': int(len(combined_nodes)),
        'total_synapses': int(len(merged_edges)),
        'circuit_roles': {k: int(v) for k, v in roles.items()},
        'motor_neuron_breakdown': {
            'front_legs_fl': int(mn_breakdown.get('fl', 0)),
            'middle_legs_ml': int(mn_breakdown.get('ml', 0)),
            'hind_legs_hl': int(mn_breakdown.get('hl', 0)),
            'wings_flight_wm': int(mn_breakdown.get('wm', 0)),
            'abdomen_ad': int(mn_breakdown.get('ad', 0)),
            'neck_head_nm': int(mn_breakdown.get('nm', 0)),
            'haltere_hm': int(mn_breakdown.get('hm', 0)),
            'special_jump_xm': int(mn_breakdown.get('xm', 0)),
            'total_motor_neurons': int(len(mn_nodes))
        },
        'neurotransmitter_distribution': {str(k): int(v) for k, v in nts.items()},
        'sensory_systems': {
            'haltere_gyroscopes': int(len(combined_nodes[combined_nodes['circuit_role'].str.contains('haltere', case=False, na=False)])),
            'visual_lptc_motion': int(len(combined_nodes[combined_nodes['circuit_role'].str.contains('visual', case=False, na=False)])),
            'proprioceptive_feco': int(len(combined_nodes[combined_nodes['circuit_role'] == 'proprioceptive_sensory'])),
            'tactile_bristles_hairplates': int(len(combined_nodes[combined_nodes['circuit_role'] == 'tactile_sensory'])),
            'mechanosensory_strain': int(len(combined_nodes[combined_nodes['circuit_role'] == 'mechanosensory']))
        },
        'descending_command_neurons': int(len(combined_nodes[combined_nodes['circuit_role'] == 'descending_neuron'])),
        'premotor_interneurons': int(len(combined_nodes[combined_nodes['circuit_role'] == 'premotor_interneuron']))
    }

    # 5. Save output files
    os.makedirs('circuits', exist_ok=True)
    out_nodes = 'circuits/comprehensive_mobility_nodes.feather'
    out_edges = 'circuits/comprehensive_mobility_edges.feather'
    out_json = 'circuits/comprehensive_mobility_summary.json'

    feather.write_feather(combined_nodes, out_nodes)
    feather.write_feather(merged_edges, out_edges)
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Saved: {out_nodes} ({os.path.getsize(out_nodes)/1024:.1f} KB)")
    print(f"[+] Saved: {out_edges} ({os.path.getsize(out_edges)/1024:.1f} KB)")
    print(f"[+] Saved: {out_json}")
    print("\nSummary:")
    print(json.dumps(summary['motor_neuron_breakdown'], indent=2))
    print("=" * 70)


if __name__ == '__main__':
    build_comprehensive_mobility_circuit()

