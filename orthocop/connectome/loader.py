"""
Connectome Data Loader and Preprocessing Pipeline for Male Drosophila CNS v1.0.
Handles loading feather tables, data validation, deduplication, annotation merging,
and exporting cleaned artifacts to data/processed/.
"""

import os
import logging
import pandas as pd
import pyarrow.feather as feather
import numpy as np

logger = logging.getLogger(__name__)

class ConnectomeLoader:
    def __init__(self, data_dir="data"):
        self.original_dir = os.path.join(data_dir, "original")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.processed_dir, exist_ok=True)
        
        self.annotations_file = os.path.join(self.original_dir, "body-annotations-male-cns-v1.0-minconf-0.5.feather")
        self.transmitters_file = os.path.join(self.original_dir, "body-neurotransmitters-male-cns-v1.0.feather")
        self.weights_file = os.path.join(self.original_dir, "connectome-weights-male-cns-v1.0-minconf-0.5.feather")
        
        self.annotations_df = None
        self.transmitters_df = None
        self.weights_df = None
        self.cleaned_annotations = None
        self.cleaned_weights = None

    def inspect_schemas(self):
        """Inspects column names, dtypes, and shapes of original files."""
        report = {}
        for name, path in [
            ("annotations", self.annotations_file),
            ("transmitters", self.transmitters_file),
            ("weights", self.weights_file)
        ]:
            if not os.path.exists(path):
                report[name] = {"error": f"File not found: {path}"}
                continue
            tbl = feather.read_table(path)
            report[name] = {
                "rows": tbl.num_rows,
                "columns": tbl.num_columns,
                "schema": {col: str(tbl.schema.field(col).type) for col in tbl.column_names}
            }
        return report

    def load_and_clean_annotations(self):
        """Loads and cleans body annotations and merges predicted neurotransmitters."""
        print("[*] Loading annotations and neurotransmitters...")
        ann_df = pd.read_feather(self.annotations_file)
        nt_df = pd.read_feather(self.transmitters_file)
        
        # Standardize body ID column name
        ann_df = ann_df.rename(columns={'bodyId': 'body_id'})
        nt_df = nt_df.rename(columns={'body': 'body_id'})
        
        initial_ann_count = len(ann_df)
        ann_df = ann_df.drop_duplicates(subset=['body_id'])
        
        # Select key neurotransmitter columns
        nt_cols_to_merge = ['body_id', 'predicted_nt', 'predicted_nt_confidence', 'consensus_nt']
        nt_sub = nt_df[nt_cols_to_merge].drop_duplicates(subset=['body_id'])
        
        # Merge
        merged_ann = pd.merge(ann_df, nt_sub, on='body_id', how='left')
        
        # Fill missing values for core categorical columns
        merged_ann['type'] = merged_ann['type'].fillna('')
        merged_ann['class'] = merged_ann['class'].fillna('')
        merged_ann['subclass'] = merged_ann['subclass'].fillna('')
        merged_ann['superclass'] = merged_ann['superclass'].fillna('')
        merged_ann['somaSide'] = merged_ann['somaSide'].fillna('unknown')
        merged_ann['somaNeuromere'] = merged_ann['somaNeuromere'].fillna('')
        merged_ann['entryNerve'] = merged_ann['entryNerve'].fillna('')
        merged_ann['exitNerve'] = merged_ann['exitNerve'].fillna('')
        merged_ann['consensus_nt'] = merged_ann['consensus_nt'].fillna(merged_ann['predicted_nt'].fillna('unknown'))
            
        # Ensure type / cell_type column is standard string
        type_col = 'type' if 'type' in merged_ann.columns else ('cell_type' if 'cell_type' in merged_ann.columns else None)
        if type_col and type_col != 'cell_type':
            merged_ann['cell_type'] = merged_ann[type_col].fillna('')
        elif 'cell_type' in merged_ann.columns:
            merged_ann['cell_type'] = merged_ann['cell_type'].fillna('')
        else:
            merged_ann['cell_type'] = ''

        # Class column
        if 'class' in merged_ann.columns:
            merged_ann['neuron_class'] = merged_ann['class'].fillna('')
        elif 'neuron_class' not in merged_ann.columns:
            merged_ann['neuron_class'] = ''

        # Hemilateral information (side / hemisphere)
        side_col = [c for c in ['side', 'hemisphere', 'nerve_cord_side'] if c in merged_ann.columns]
        if side_col:
            merged_ann['side'] = merged_ann[side_col[0]].fillna('unknown')
        else:
            # Infer from name/type suffix (_L / _R) if present
            merged_ann['side'] = 'unknown'
            if 'cell_type' in merged_ann.columns:
                merged_ann.loc[merged_ann['cell_type'].str.endswith('_L', na=False), 'side'] = 'L'
                merged_ann.loc[merged_ann['cell_type'].str.endswith('_R', na=False), 'side'] = 'R'

        self.cleaned_annotations = merged_ann
        out_path = os.path.join(self.processed_dir, "cleaned_annotations.feather")
        merged_ann.to_feather(out_path)
        print(f"[+] Cleaned annotations saved: {len(merged_ann):,} neurons (deduped from {initial_ann_count:,}) to {out_path}")
        return merged_ann

    def load_and_clean_weights(self, min_weight=1):
        """Loads and cleans synaptic connectivity weights table."""
        print(f"[*] Loading connectivity weights (filtering weight >= {min_weight})...")
        weights_df = pd.read_feather(self.weights_file)
        
        # Standardize column names
        pre_col = [c for c in ['body_pre', 'pre', 'bodyId_pre', 'PreSyn'] if c in weights_df.columns][0]
        post_col = [c for c in ['body_post', 'post', 'bodyId_post', 'PostSyn'] if c in weights_df.columns][0]
        weight_col = [c for c in ['weight', 'count', 'syn_count', 'synapses'] if c in weights_df.columns][0]
        
        weights_df = weights_df.rename(columns={
            pre_col: 'pre_id',
            post_col: 'post_id',
            weight_col: 'weight'
        })
        
        init_edges = len(weights_df)
        # Remove zero or negative weights
        weights_df = weights_df[weights_df['weight'] >= min_weight]
        # Aggregate multiple entries for same pre->post pair if any
        weights_df = weights_df.groupby(['pre_id', 'post_id'], as_index=False)['weight'].sum()
        
        self.cleaned_weights = weights_df
        out_path = os.path.join(self.processed_dir, "cleaned_weights.feather")
        weights_df.to_feather(out_path)
        print(f"[+] Cleaned connectivity weights saved: {len(weights_df):,} edges (from {init_edges:,}) to {out_path}")
        return weights_df

if __name__ == "__main__":
    loader = ConnectomeLoader()
    schema = loader.inspect_schemas()
    for k, v in schema.items():
        print(k, ":", v)
