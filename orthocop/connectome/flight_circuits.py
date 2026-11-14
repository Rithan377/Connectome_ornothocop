"""
Identification and extraction of flight-related neural circuits from Male CNS v1.0.
Combines anatomical annotations, nerve pathways, cell-type labels, neurotransmitters,
and connectivity evidence to isolate the flight sensory-to-motor network.
"""

import os
import json
import numpy as np
import pandas as pd
import pyarrow.feather as feather
import pyarrow.compute as pc
import pyarrow as pa

class FlightCircuitExtractor:
    def __init__(self, data_dir="data", output_dir="circuits"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.ann_path = os.path.join(data_dir, "processed", "cleaned_annotations.feather")
        self.weights_path = os.path.join(data_dir, "original", "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
        
        self.annotations = pd.read_feather(self.ann_path)
        self.weights_table = None

    def load_weights_table(self):
        if self.weights_table is None:
            print("[*] Memory-mapping full weights feather table...")
            self.weights_table = feather.read_table(self.weights_path, memory_map=True)
        return self.weights_table

    def identify_flight_candidate_neurons(self, min_dn_synapses_to_mn=50):
        """
        Identifies candidate flight-related neurons across 4 key populations:
        1. Motor neurons: Wing & power muscles (subclass wm) + T2 vnc_motor
        2. Sensory neurons: Haltere afferents (DMetaN), Wing afferents (ADMN, PDMN), Visual LPTCs (VS, HS)
        3. Descending neurons (DNs): Direct drivers of flight motor neurons with weight >= threshold
        4. Premotor / local interneurons: T2/T3 VNC interneurons mediating strong recurrent flight control
        """
        ann = self.annotations
        
        # 1. Flight Motor Neurons
        # Wing steering muscles (b1, b2, i1, i2, etc.) and power muscles (DLM, DVM)
        flight_mns = ann[
            (ann['subclass'] == 'wm') | 
            (ann['type'].str.contains('MN|DLM|DVM', case=False, na=False) & (ann['somaNeuromere'] == 'T2'))
        ].copy()
        flight_mns['circuit_role'] = 'motor_neuron'
        
        # 2. Sensory Afferents
        # Haltere: DMetaN (Dorsal Metathoracic Nerve)
        haltere_sensory = ann[ann['entryNerve'] == 'DMetaN'].copy()
        haltere_sensory['circuit_role'] = 'haltere_sensory'
        
        # Wing mechanosensory: ADMN and PDMN
        wing_sensory = ann[ann['entryNerve'].isin(['ADMN', 'PDMN']) & (ann['superclass'] == 'vnc_sensory')].copy()
        wing_sensory['circuit_role'] = 'wing_sensory'
        
        # Visual LPTCs: VS, HS tangential cells
        vis_mask1 = ann['type'].str.contains(r'^(?:VS|HS|CH|H1|H2)', regex=True, na=False)
        vis_mask2 = (ann['class'] == 'visual') & ann['type'].str.contains('tangential|lobula', case=False, na=False)
        visual_sensory = ann[vis_mask1 | vis_mask2].copy()
        visual_sensory['circuit_role'] = 'visual_sensory'
        
        mn_ids = set(flight_mns['body_id'])
        print(f"[+] Identified {len(flight_mns)} flight MNs, {len(haltere_sensory)} haltere afferents, {len(wing_sensory)} wing afferents, {len(visual_sensory)} visual LPTCs.")
        
        # 3. Descending Neurons connecting to flight MNs
        tbl = self.load_weights_table()
        mn_pa_set = pa.array(list(mn_ids), type=pa.int64())
        mask_mn = pc.is_in(tbl['body_post'], value_set=mn_pa_set)
        up_to_mn = tbl.filter(mask_mn).to_pandas()
        
        # Group by pre_id to calculate total connection strength to flight MNs
        dn_all_ids = set(ann[ann['superclass'] == 'descending_neuron']['body_id'])
        dn_weights = up_to_mn[up_to_mn['body_pre'].isin(dn_all_ids)].groupby('body_pre')['weight'].sum()
        selected_dn_ids = set(dn_weights[dn_weights >= min_dn_synapses_to_mn].index)
        
        descending_neurons = ann[ann['body_id'].isin(selected_dn_ids)].copy()
        descending_neurons['circuit_role'] = 'descending_neuron'
        print(f"[+] Identified {len(descending_neurons)} major flight-associated Descending Neurons (synapses >= {min_dn_synapses_to_mn}).")
        
        # 4. Premotor Interneurons in T2/T3 with strong connectivity to MNs and from Haltere/DNs
        vnc_intr_ids = set(ann[ann['superclass'] == 'vnc_intrinsic']['body_id'])
        vnc_pre_weights = up_to_mn[up_to_mn['body_pre'].isin(vnc_intr_ids)].groupby('body_pre')['weight'].sum()
        # Top 150 strongest premotor interneurons
        top_premotor_ids = set(vnc_pre_weights.sort_values(ascending=False).head(150).index)
        premotor_neurons = ann[ann['body_id'].isin(top_premotor_ids)].copy()
        premotor_neurons['circuit_role'] = 'premotor_interneuron'
        print(f"[+] Identified {len(premotor_neurons)} top flight premotor interneurons.")
        
        # Combine all candidates
        all_candidates = pd.concat([
            flight_mns,
            haltere_sensory,
            wing_sensory,
            visual_sensory,
            descending_neurons,
            premotor_neurons
        ], ignore_index=True).drop_duplicates(subset=['body_id'])
        
        # Save catalogue
        cat_file = os.path.join(self.output_dir, "flight_candidate_neurons.feather")
        all_candidates.to_feather(cat_file)
        
        summary = {
            "total_candidates": len(all_candidates),
            "breakdown": all_candidates['circuit_role'].value_counts().to_dict(),
            "neurotransmitter_breakdown": all_candidates['consensus_nt'].value_counts().to_dict(),
            "hemisphere_breakdown": all_candidates['somaSide'].value_counts().to_dict()
        }
        with open(os.path.join(self.output_dir, "flight_candidate_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
            
        print(f"[+] Flight candidate catalogue saved ({len(all_candidates):,} neurons) to {cat_file}")
        return all_candidates

    def extract_subnetwork(self, candidate_df=None, min_synapse_threshold=2):
        """
        Extracts all directed connections between identified candidate neurons
        with weight >= min_synapse_threshold.
        """
        if candidate_df is None:
            cat_path = os.path.join(self.output_dir, "flight_candidate_neurons.feather")
            if os.path.exists(cat_path):
                candidate_df = pd.read_feather(cat_path)
            else:
                candidate_df = self.identify_flight_candidate_neurons()
                
        candidate_ids = set(candidate_df['body_id'])
        print(f"[*] Extracting subnetwork among {len(candidate_ids):,} candidate neurons (weight >= {min_synapse_threshold})...")
        
        tbl = self.load_weights_table()
        pa_ids = pa.array(list(candidate_ids), type=pa.int64())
        
        mask_pre = pc.is_in(tbl['body_pre'], value_set=pa_ids)
        mask_post = pc.is_in(tbl['body_post'], value_set=pa_ids)
        mask_both = pc.and_(mask_pre, mask_post)
        
        sub_tbl = tbl.filter(mask_both)
        df_sub = sub_tbl.to_pandas()
        
        # Filter by threshold
        df_sub = df_sub[df_sub['weight'] >= min_synapse_threshold].copy()
        
        # Standardize column names
        df_sub = df_sub.rename(columns={'body_pre': 'pre_id', 'body_post': 'post_id'})
        
        # Save subnetwork edges
        out_feather = os.path.join(self.output_dir, f"flight_subnetwork_edges_min{min_synapse_threshold}.feather")
        df_sub.to_feather(out_feather)
        
        print(f"[+] Subnetwork extracted: {len(df_sub):,} edges among candidate neurons. Saved to {out_feather}")
        return df_sub, candidate_df

if __name__ == "__main__":
    extractor = FlightCircuitExtractor()
    candidates = extractor.identify_flight_candidate_neurons()
    edges, _ = extractor.extract_subnetwork(candidates, min_synapse_threshold=2)
