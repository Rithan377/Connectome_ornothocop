"""
Build Graphify Knowledge Graph for OrthoCop Project.
Generates an interactive HTML graph visualization, community clusters,
and GRAPH_REPORT.md audit report for all codebase modules, circuits, and experiments.
"""

import os
import sys
import json
from pathlib import Path

def main():
    print("[*] Running Graphify Knowledge Graph Generator...")
    out_dir = Path("graphify-out")
    out_dir.mkdir(exist_ok=True)
    
    python_bin = sys.executable
    (out_dir / ".graphify_python").write_text(python_bin, encoding="utf-8")
    (out_dir / ".graphify_root").write_text(str(Path(".").resolve()), encoding="utf-8")
    
    from graphify.detect import detect
    detect_res = detect(Path("."))
    (out_dir / ".graphify_detect.json").write_text(json.dumps(detect_res, ensure_ascii=False), encoding="utf-8")
    print(f"[*] Detected {detect_res.get('total_files', 0)} files (~{detect_res.get('total_words', 0):,} words).")
    
    # Structural extraction for code
    from graphify.extract import collect_files, extract
    code_files = []
    for f in detect_res.get('files', {}).get('code', []):
        p = Path(f)
        # Exclude virtual environments or large binary folders
        if "myenv" in str(p) or ".git" in str(p):
            continue
        code_files.extend(collect_files(p) if p.is_dir() else [p])
        
    print(f"[*] Extracting AST from {len(code_files)} code files...")
    ast_res = extract(code_files, cache_root=Path("."))
    (out_dir / ".graphify_ast.json").write_text(json.dumps(ast_res, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Empty semantic file since code is extracted via AST
    (out_dir / ".graphify_semantic.json").write_text(json.dumps({'nodes':[], 'edges':[], 'hyperedges':[], 'input_tokens':0, 'output_tokens':0}), encoding='utf-8')
    
    # Merge AST + semantic
    merged = {
        'nodes': ast_res['nodes'],
        'edges': ast_res['edges'],
        'hyperedges': [],
        'input_tokens': 0,
        'output_tokens': 0
    }
    (out_dir / ".graphify_extract.json").write_text(json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Build Network
    from graphify.build import build_from_json
    from graphify.cluster import cluster, score_all
    from graphify.analyze import god_nodes, surprising_connections, suggest_questions
    from graphify.report import generate
    from graphify.export import to_json
    
    G = build_from_json(merged, root='.', directed=True)
    if G.number_of_nodes() == 0:
        print("[!] Warning: graph has 0 nodes.")
        return
        
    communities = cluster(G)
    cohesion = score_all(G, communities)
    gods = god_nodes(G)
    surprises = surprising_connections(G, communities)
    labels = {cid: f"Community {cid}" for cid in communities}
    questions = suggest_questions(G, communities, labels)
    
    to_json(G, communities, str(out_dir / "graph.json"))
    report = generate(G, communities, cohesion, labels, gods, surprises, detect_res, {'input': 0, 'output': 0}, '.', suggested_questions=questions)
    (out_dir / "GRAPH_REPORT.md").write_text(report, encoding="utf-8")
    
    print(f"[+] Knowledge Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges, {len(communities)} communities.")
    print(f"[+] Graph report saved to {out_dir / 'GRAPH_REPORT.md'}")
    
    # Generate HTML visualization
    try:
        from graphify.export import to_html
        to_html(G, communities, str(out_dir / "graph.html"))
        print(f"[+] Interactive HTML graph visualization generated at {out_dir / 'graph.html'}")
    except Exception as e:
        # Fallback using graphify export command if function name differs
        os.system(f"{python_bin} -m graphify.cli export html 2>/dev/null || true")

if __name__ == "__main__":
    main()
