"""
Build Stabilized Flight Circuit.

Combines:
- 74 sensory neurons from the 148-circuit (haltere, visual, wing) with correct circuit_role labels
- 24 descending neurons from the 148-circuit
- All 62 wing flight motor neurons from the full motor connectome (23 muscle types)
- All 737 premotor interneurons that directly synapse onto flight MNs (247 GABAergic inhibitory)
- Edges between all these neurons from both circuit files

This gives ~897 neurons with full biological inhibitory damping against 200 Hz wing-beat resonance.
"""

import pandas as pd
import pyarrow.feather as feather
import numpy as np
import os


def build_stabilized_flight_circuit():
    """Build and save the stabilized flight circuit."""
    # Load source data
    nodes_148 = feather.read_table('circuits/ultra_minimal_flight_nodes.feather').to_pandas()
    nodes_full = feather.read_table('circuits/motor_connectome_nodes.feather').to_pandas()
    edges_full = feather.read_table('circuits/motor_connectome_edges.feather').to_pandas()
    edges_148 = feather.read_table('circuits/ultra_minimal_flight_edges.feather').to_pandas()

    # 1. Sensory neurons from 148-circuit (preserve circuit_role labels)
    sensory_roles = {'haltere_sensory', 'visual_sensory', 'wing_sensory'}
    sensory_148 = nodes_148[nodes_148['circuit_role'].isin(sensory_roles)]
    sensory_ids = set(sensory_148['body_id'].tolist())
    print(f"  Sensory from 148-circuit: {len(sensory_ids)}")

    # 2. Descending neurons from 148-circuit
    dn_148 = nodes_148[nodes_148['circuit_role'] == 'descending_neuron']
    dn_ids = set(dn_148['body_id'].tolist())
    print(f"  DNs from 148-circuit: {len(dn_ids)}")

    # 3. ALL wing flight motor neurons from full connectome (23 types)
    flight_mn_patterns = [
        'DLMn', 'DVMn', 'b1 MN', 'b2 MN', 'i1 MN', 'i2 MN',
        'hg1 MN', 'hg2 MN', 'hg3 MN', 'hg4 MN',
        'ps1 MN', 'ps2 MN', 'tp1 MN', 'tp2 MN', 'tpn MN',
        'hi1 MN', 'hi2 MN', 'hiii2 MN', 'iii1 MN', 'hDVM MN'
    ]
    pattern = '|'.join(flight_mn_patterns)
    flight_mns = nodes_full[
        (nodes_full['circuit_role'] == 'motor_neuron') &
        nodes_full['type'].str.contains(pattern, case=False, na=False)
    ]
    flight_mn_ids = set(flight_mns['body_id'].tolist())
    print(f"  Flight MNs from full connectome: {len(flight_mn_ids)}")
    print(f"    Types: {flight_mns['type'].value_counts().to_dict()}")

    # 4. Premotor interneurons that DIRECTLY synapse onto flight MNs
    pre_to_flight = edges_full[edges_full['post_id'].isin(flight_mn_ids)]
    premotor_candidate_ids = set(pre_to_flight['pre_id'].tolist()) - flight_mn_ids
    premotor_nodes = nodes_full[
        (nodes_full['body_id'].isin(premotor_candidate_ids)) &
        (nodes_full['circuit_role'] == 'premotor_interneuron')
    ]
    premotor_ids = set(premotor_nodes['body_id'].tolist())
    gaba_count = len(premotor_nodes[premotor_nodes['consensus_nt'] == 'gaba'])
    glut_count = len(premotor_nodes[premotor_nodes['consensus_nt'] == 'glutamate'])
    print(f"  Premotor INs (direct→flight MNs): {len(premotor_ids)}")
    print(f"    GABAergic (inhibitory): {gaba_count}")
    print(f"    Glutamatergic (inhibitory): {glut_count}")
    print(f"    Cholinergic (excitatory): {len(premotor_ids) - gaba_count - glut_count}")

    # 5. Combine node tables
    from_148_ids = sensory_ids | dn_ids
    from_full_ids = (flight_mn_ids | premotor_ids) - from_148_ids

    nodes_from_148 = nodes_148[nodes_148['body_id'].isin(from_148_ids)].copy()
    nodes_from_full = nodes_full[nodes_full['body_id'].isin(from_full_ids)].copy()

    # Align columns (take intersection)
    common_cols = sorted(set(nodes_from_148.columns) & set(nodes_from_full.columns))
    combined_nodes = pd.concat(
        [nodes_from_148[common_cols], nodes_from_full[common_cols]],
        ignore_index=True
    ).drop_duplicates(subset='body_id', keep='first')

    # 6. Extract edges where both endpoints are in our circuit
    all_body_ids = set(combined_nodes['body_id'].tolist())

    edges_from_full = edges_full[
        edges_full['pre_id'].isin(all_body_ids) &
        edges_full['post_id'].isin(all_body_ids)
    ][['pre_id', 'post_id', 'weight']].copy()

    edges_from_148 = edges_148[
        edges_148['pre_id'].isin(all_body_ids) &
        edges_148['post_id'].isin(all_body_ids)
    ][['pre_id', 'post_id', 'weight']].copy()

    # Merge and deduplicate (keep max weight for duplicate edges)
    all_edges = pd.concat([edges_from_full, edges_from_148], ignore_index=True)
    combined_edges = all_edges.groupby(['pre_id', 'post_id'], as_index=False)['weight'].max()

    # Summary
    print(f"\n  === Stabilized Flight Circuit ===")
    print(f"  Total neurons: {len(combined_nodes)}")
    print(f"  Total edges:   {len(combined_edges)}")
    print(f"  Role breakdown: {combined_nodes['circuit_role'].value_counts().to_dict()}")
    nt_dist = combined_nodes['consensus_nt'].value_counts().to_dict()
    print(f"  NT breakdown:   {nt_dist}")

    # Save
    out_nodes = 'circuits/stabilized_flight_nodes.feather'
    out_edges = 'circuits/stabilized_flight_edges.feather'
    feather.write_feather(combined_nodes.reset_index(drop=True), out_nodes)
    feather.write_feather(combined_edges.reset_index(drop=True), out_edges)
    print(f"\n  Saved: {out_nodes}")
    print(f"  Saved: {out_edges}")

    return len(combined_nodes), len(combined_edges)


if __name__ == '__main__':
    build_stabilized_flight_circuit()
