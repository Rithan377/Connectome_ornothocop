"""
Graph-theoretic analysis of the Drosophila Male Connectome flight subnetwork using NetworkX.
Implements degree, strength, betweenness centrality, shortest path, clustering,
strongly connected components, bilateral asymmetry, and threshold sensitivity.
"""

import os
import json
import networkx as nx
import numpy as np
import pandas as pd

class FlightGraphAnalyzer:
    def __init__(self, edges_df, nodes_df):
        self.edges_df = edges_df.copy()
        self.nodes_df = nodes_df.copy()
        self.node_meta = self.nodes_df.set_index('body_id').to_dict('index')
        self.G = self.build_graph()

    def build_graph(self):
        """Builds a directed weighted NetworkX graph G=(V, E)."""
        G = nx.DiGraph()
        # Add nodes with metadata
        for body_id, meta in self.node_meta.items():
            G.add_node(
                body_id,
                type=meta.get('type', ''),
                circuit_role=meta.get('circuit_role', 'unknown'),
                somaSide=meta.get('somaSide', 'unknown'),
                nt=meta.get('consensus_nt', 'unknown')
            )
        # Add edges with weights and distances
        for _, row in self.edges_df.iterrows():
            u = int(row['pre_id'])
            v = int(row['post_id'])
            w = float(row['weight'])
            if G.has_node(u) and G.has_node(v):
                # distance inverse to connection weight
                G.add_edge(u, v, weight=w, distance=1.0 / w)
        return G

    def compute_centrality_metrics(self):
        """Computes in-degree, out-degree, strengths, and betweenness centrality."""
        G = self.G
        in_degrees = dict(G.in_degree())
        out_degrees = dict(G.out_degree())
        in_strengths = dict(G.in_degree(weight='weight'))
        out_strengths = dict(G.out_degree(weight='weight'))
        
        print("[*] Calculating betweenness centrality (weighted)...")
        betweenness = nx.betweenness_centrality(G, weight='distance', normalized=True)
        
        metrics = []
        for n in G.nodes():
            meta = self.node_meta.get(n, {})
            metrics.append({
                'body_id': n,
                'type': meta.get('type', ''),
                'circuit_role': meta.get('circuit_role', ''),
                'side': meta.get('somaSide', ''),
                'nt': meta.get('consensus_nt', ''),
                'in_degree': in_degrees.get(n, 0),
                'out_degree': out_degrees.get(n, 0),
                'in_strength': in_strengths.get(n, 0.0),
                'out_strength': out_strengths.get(n, 0.0),
                'betweenness_centrality': betweenness.get(n, 0.0)
            })
        df_metrics = pd.DataFrame(metrics)
        return df_metrics

    def analyze_recurrent_components(self):
        """Finds strongly connected components and local clustering coefficients."""
        G = self.G
        sccs = list(nx.strongly_connected_components(G))
        # Filter non-trivial components with size > 1
        recurrent_sccs = [c for c in sccs if len(c) > 1]
        recurrent_sccs.sort(key=len, reverse=True)
        
        # Clustering coefficient on underlying undirected representation
        clustering = nx.clustering(G.to_undirected(), weight='weight')
        mean_clustering = np.mean(list(clustering.values())) if clustering else 0.0
        
        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "total_sccs": len(sccs),
            "recurrent_sccs_count": len(recurrent_sccs),
            "largest_scc_size": len(recurrent_sccs[0]) if recurrent_sccs else 0,
            "mean_clustering_coefficient": float(mean_clustering)
        }

    def compute_shortest_paths_sensory_to_motor(self, max_paths_per_pair=1):
        """Finds shortest structural pathways from sensory afferents to motor neurons."""
        G = self.G
        sensory_nodes = [n for n, d in G.nodes(data=True) if 'sensory' in d.get('circuit_role', '')]
        motor_nodes = [n for n, d in G.nodes(data=True) if d.get('circuit_role') == 'motor_neuron']
        
        path_results = []
        for s in sensory_nodes[:30]: # Sample prominent sensory afferents
            for m in motor_nodes[:20]: # Sample prominent motor neurons
                if nx.has_path(G, s, m):
                    path = nx.shortest_path(G, source=s, target=m, weight='distance')
                    cost = nx.shortest_path_length(G, source=s, target=m, weight='distance')
                    path_results.append({
                        'source': s,
                        'source_type': G.nodes[s].get('type'),
                        'source_role': G.nodes[s].get('circuit_role'),
                        'target': m,
                        'target_type': G.nodes[m].get('type'),
                        'path_hops': len(path) - 1,
                        'path_cost': float(cost),
                        'path': [int(node) for node in path]
                    })
        return pd.DataFrame(path_results)

    def bilateral_connectivity_analysis(self):
        """Calculates bilateral difference: Delta W_LR = W_L - W_R across homologous cell types."""
        G = self.G
        # Match neurons by (type, circuit_role) between left and right sides
        df_nodes = self.nodes_df[self.nodes_df['somaSide'].isin(['L', 'R', 'left', 'right'])].copy()
        df_nodes['clean_side'] = df_nodes['somaSide'].apply(lambda x: 'L' if x.startswith('L') or x == 'left' else 'R')
        
        # Calculate strengths for L and R neurons of same cell types
        types = df_nodes['type'].unique()
        bilateral_diffs = []
        
        in_strengths = dict(G.in_degree(weight='weight'))
        out_strengths = dict(G.out_degree(weight='weight'))
        
        for t in types:
            if not t or t == '':
                continue
            left_neurons = df_nodes[(df_nodes['type'] == t) & (df_nodes['clean_side'] == 'L')]['body_id'].tolist()
            right_neurons = df_nodes[(df_nodes['type'] == t) & (df_nodes['clean_side'] == 'R')]['body_id'].tolist()
            if left_neurons and right_neurons:
                wl_in = np.mean([in_strengths.get(n, 0) for n in left_neurons])
                wr_in = np.mean([in_strengths.get(n, 0) for n in right_neurons])
                wl_out = np.mean([out_strengths.get(n, 0) for n in left_neurons])
                wr_out = np.mean([out_strengths.get(n, 0) for n in right_neurons])
                
                delta_w_in = wl_in - wr_in
                delta_w_out = wl_out - wr_out
                asymmetry_index = (abs(wl_in - wr_in) + abs(wl_out - wr_out)) / (wl_in + wr_in + wl_out + wr_out + 1e-6)
                
                bilateral_diffs.append({
                    'type': t,
                    'left_count': len(left_neurons),
                    'right_count': len(right_neurons),
                    'wl_in': float(wl_in),
                    'wr_in': float(wr_in),
                    'delta_w_in': float(delta_w_in),
                    'wl_out': float(wl_out),
                    'wr_out': float(wr_out),
                    'delta_w_out': float(delta_w_out),
                    'asymmetry_index': float(asymmetry_index)
                })
        return pd.DataFrame(bilateral_diffs)

def run_threshold_sensitivity(edges_raw, nodes_df, thresholds=[1, 2, 3, 5, 10, 20]):
    """Calculates network properties across varying connectivity thresholds."""
    results = []
    for thr in thresholds:
        sub_edges = edges_raw[edges_raw['weight'] >= thr]
        analyzer = FlightGraphAnalyzer(sub_edges, nodes_df)
        G = analyzer.G
        
        # Check sensory to motor reachability
        sensory_nodes = [n for n, d in G.nodes(data=True) if 'sensory' in d.get('circuit_role', '')]
        motor_nodes = [n for n, d in G.nodes(data=True) if d.get('circuit_role') == 'motor_neuron']
        
        reachable_pairs = 0
        total_tested = min(50, len(sensory_nodes)) * min(20, len(motor_nodes))
        for s in sensory_nodes[:50]:
            for m in motor_nodes[:20]:
                if nx.has_path(G, s, m):
                    reachable_pairs += 1
                    
        reachability_ratio = reachable_pairs / max(1, total_tested)
        results.append({
            'threshold': thr,
            'active_edges': G.number_of_edges(),
            'active_nodes_with_edges': len([n for n, d in G.degree() if d > 0]),
            'mean_degree': np.mean([d for _, d in G.degree()]),
            'reachability_sensory_to_motor': reachability_ratio
        })
    return pd.DataFrame(results)

if __name__ == "__main__":
    nodes_df = pd.read_feather("circuits/flight_candidate_neurons.feather")
    edges_df = pd.read_feather("circuits/flight_subnetwork_edges_min2.feather")
    
    print("[*] Running Flight Graph Analysis...")
    analyzer = FlightGraphAnalyzer(edges_df, nodes_df)
    
    metrics_df = analyzer.compute_centrality_metrics()
    metrics_df.to_feather("circuits/graph_centrality_metrics.feather")
    print(f"[+] Centrality metrics computed for {len(metrics_df):,} nodes.")
    print("Top 10 nodes by Betweenness Centrality:")
    print(metrics_df.sort_values(by='betweenness_centrality', ascending=False)[['body_id', 'type', 'circuit_role', 'betweenness_centrality', 'in_strength', 'out_strength']].head(10))
    
    recurrent = analyzer.analyze_recurrent_components()
    print("\nRecurrent organization:", json.dumps(recurrent, indent=2))
    
    bilateral_df = analyzer.bilateral_connectivity_analysis()
    bilateral_df.to_feather("circuits/bilateral_asymmetry.feather")
    print("\nTop 5 Most Asymmetric Flight Types:")
    print(bilateral_df.sort_values('asymmetry_index', ascending=False)[['type', 'asymmetry_index', 'delta_w_in', 'delta_w_out']].head(5))
    
    sens_df = run_threshold_sensitivity(edges_df, nodes_df, thresholds=[2, 3, 5, 10, 20])
    sens_df.to_feather("circuits/threshold_sensitivity.feather")
    print("\nThreshold Sensitivity Summary:")
    print(sens_df)
