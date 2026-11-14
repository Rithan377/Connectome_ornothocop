"""
Builds the complete Interactive 3D Virtual Fly Flight Arena.
Embeds the Janelia/DeepMind FlyBody 3D morphology with the embedded 3D connectome brain
(Full Candidate 2,300-Neuron vs Ultra-Reduced 148-Neuron Minimal Core).
Generates:
1. visualizations/virtual_fly_flight_arena.html (Complete standalone interactive 3D WebGL simulator)
2. Copies to artifacts for direct user access.
"""

import os
import json
import numpy as np
import pandas as pd
from orthocop.visualization.build_3d_visualizer import prepare_3d_coordinates

def parse_obj_unique(path, max_faces=None):
    """Parses an OBJ file, extracts unique vertices, and indexes faces."""
    verts = []
    with open(path) as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.strip().split()[1:4]])
    verts = np.array(verts, dtype=np.float32)
    u_verts, inv = np.unique(verts, axis=0, return_inverse=True)
    
    if max_faces is not None and len(inv) // 3 > max_faces:
        step = int(np.ceil((len(inv) // 3) / max_faces))
        inv = inv.reshape(-1, 3)[::step].flatten()
        active_u = np.unique(inv)
        remap = {old: new for new, old in enumerate(active_u)}
        inv = np.array([remap[x] for x in inv], dtype=np.int32)
        u_verts = u_verts[active_u]
        
    return u_verts, inv

def to_threejs_coords(verts_array, center_offset=np.array([0.0, 0.0, 1.18])):
    """
    Transforms FlyBody coordinates (X forward, Y lateral, Z up)
    to Three.js coordinates (X lateral, Y up, Z backward/forward).
    """
    v = verts_array - center_offset
    three_x = v[:, 1]
    three_y = v[:, 2]
    three_z = -v[:, 0]
    return np.column_stack([three_x, three_y, three_z])

def build_virtual_fly_arena(
    output_html="visualizations/virtual_fly_flight_arena.html",
    asset_dir="visualizations/fly_assets",
    circuits_dir="circuits",
    results_dir="results"
):
    print("[*] Extracting 3D FlyBody Anatomical Meshes...")
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    
    # 1. Process FlyBody meshes
    meshes = {}
    
    # Head
    head_v, head_i = parse_obj_unique(os.path.join(asset_dir, 'head_body.obj'))
    head_three = to_threejs_coords(head_v)
    meshes['head'] = {'v': np.round(head_three, 4).tolist(), 'i': head_i.tolist()}
    
    # Compound Eyes (head_red)
    eyes_v, eyes_i = parse_obj_unique(os.path.join(asset_dir, 'head_red.obj'), max_faces=12000)
    eyes_three = to_threejs_coords(eyes_v)
    meshes['eyes'] = {'v': np.round(eyes_three, 4).tolist(), 'i': eyes_i.tolist()}
    
    # Thorax
    thorax_v, thorax_i = parse_obj_unique(os.path.join(asset_dir, 'thorax_body.obj'))
    thorax_three = to_threejs_coords(thorax_v)
    meshes['thorax'] = {'v': np.round(thorax_three, 4).tolist(), 'i': thorax_i.tolist()}
    
    # Left & Right Wings (centered at wing hinges)
    l_hinge = np.array([0.0484, -0.4619, 1.3266])
    r_hinge = np.array([0.0484, 0.4619, 1.3266])
    
    wl_v, wl_i = parse_obj_unique(os.path.join(asset_dir, 'wing_left.obj'))
    # Local vertices relative to hinge
    wl_local = wl_v - l_hinge
    # Transform local coordinate orientation
    wl_three = np.column_stack([wl_local[:, 1], wl_local[:, 2], -wl_local[:, 0]])
    meshes['wing_left'] = {
        'v': np.round(wl_three, 4).tolist(),
        'i': wl_i.tolist(),
        'hinge': np.round(to_threejs_coords(l_hinge.reshape(1, -1))[0], 4).tolist()
    }
    
    wr_v, wr_i = parse_obj_unique(os.path.join(asset_dir, 'wing_right.obj'))
    wr_local = wr_v - r_hinge
    wr_three = np.column_stack([wr_local[:, 1], wr_local[:, 2], -wr_local[:, 0]])
    meshes['wing_right'] = {
        'v': np.round(wr_three, 4).tolist(),
        'i': wr_i.tolist(),
        'hinge': np.round(to_threejs_coords(r_hinge.reshape(1, -1))[0], 4).tolist()
    }
    
    # Halteres (centered at haltere hinges)
    hl_hinge = np.array([0.3245, -0.2393, 1.1634])
    hr_hinge = np.array([0.3245, 0.2393, 1.1634])
    
    hl_v, hl_i = parse_obj_unique(os.path.join(asset_dir, 'haltere_left.obj'))
    hl_local = hl_v - hl_hinge
    hl_three = np.column_stack([hl_local[:, 1], hl_local[:, 2], -hl_local[:, 0]])
    meshes['haltere_left'] = {
        'v': np.round(hl_three, 4).tolist(),
        'i': hl_i.tolist(),
        'hinge': np.round(to_threejs_coords(hl_hinge.reshape(1, -1))[0], 4).tolist()
    }
    
    hr_v, hr_i = parse_obj_unique(os.path.join(asset_dir, 'haltere_right.obj'))
    hr_local = hr_v - hr_hinge
    hr_three = np.column_stack([hr_local[:, 1], hr_local[:, 2], -hr_local[:, 0]])
    meshes['haltere_right'] = {
        'v': np.round(hr_three, 4).tolist(),
        'i': hr_i.tolist(),
        'hinge': np.round(to_threejs_coords(hr_hinge.reshape(1, -1))[0], 4).tolist()
    }
    
    # Abdomen Segments 1..8 combined
    all_abd_v = []
    all_abd_i = []
    v_off = 0
    for i in range(1, 9):
        p = os.path.join(asset_dir, f'abdomen_{i}.obj')
        if os.path.exists(p):
            av, ai = parse_obj_unique(p, max_faces=1500)
            all_abd_v.extend(av)
            all_abd_i.extend([x + v_off for x in ai])
            v_off += len(av)
    abd_three = to_threejs_coords(np.array(all_abd_v, dtype=np.float32))
    meshes['abdomen'] = {'v': np.round(abd_three, 4).tolist(), 'i': all_abd_i}
    
    # 2. Extract Connectome Brains (Full vs Ultra-Reduced)
    print("[*] Processing 3D Connectome Neurons and Synaptic Pathways...")
    full_nodes = pd.read_feather(os.path.join(circuits_dir, "flight_candidate_neurons.feather"))
    full_edges = pd.read_feather(os.path.join(circuits_dir, "flight_subnetwork_edges_min2.feather"))
    
    min_nodes = pd.read_feather(os.path.join(circuits_dir, "ultra_minimal_flight_nodes.feather"))
    min_edges = pd.read_feather(os.path.join(circuits_dir, "ultra_minimal_flight_edges.feather"))
    
    full_3d = prepare_3d_coordinates(full_nodes)
    min_3d = prepare_3d_coordinates(min_nodes)
    
    def transform_brain_coords(df):
        z_conn = df['z_um'].values
        x_conn = df['x_um'].values
        y_conn = df['y_um'].values
        
        # AP: connectome z -> flybody x
        fly_x = -0.85 + (z_conn - 100.0) / 900.0 * 1.15
        # Lateral: connectome x -> flybody y
        fly_y = (x_conn - 390.0) / 200.0 * 0.28
        # DV: connectome y -> flybody z
        fly_z = 1.15 - (y_conn - 350.0) / 200.0 * 0.15
        
        verts = np.column_stack([fly_x, fly_y, fly_z])
        three_pts = to_threejs_coords(verts)
        return three_pts
    
    full_pts = transform_brain_coords(full_3d)
    min_pts = transform_brain_coords(min_3d)
    
    full_3d['tx'] = full_pts[:, 0]
    full_3d['ty'] = full_pts[:, 1]
    full_3d['tz'] = full_pts[:, 2]
    
    min_3d['tx'] = min_pts[:, 0]
    min_3d['ty'] = min_pts[:, 1]
    min_3d['tz'] = min_pts[:, 2]
    
    role_colors = {
        'haltere_sensory': '#ff3b30',      # Crimson Red
        'wing_sensory': '#ff9500',         # Amber
        'visual_sensory': '#00f2fe',       # Cyan
        'descending_neuron': '#007aff',    # Royal Blue
        'premotor_interneuron': '#af52de', # Purple
        'motor_neuron': '#34c759'          # Emerald Green
    }
    
    def pack_brain(df, edges_df, max_edges=1200):
        c_dict = df.set_index('body_id')[['tx', 'ty', 'tz']].to_dict('index')
        nodes_list = []
        for _, row in df.iterrows():
            bid = int(row['body_id'])
            nodes_list.append({
                'id': bid,
                'pos': [round(row['tx'], 4), round(row['ty'], 4), round(row['tz'], 4)],
                'role': row['circuit_role'],
                'color': role_colors.get(row['circuit_role'], '#ffffff'),
                'type': str(row.get('type', '')),
                'nt': str(row.get('consensus_nt', 'ACH'))
            })
        edges_list = []
        for _, row in edges_df.head(max_edges).iterrows():
            u = int(row['pre_id'])
            v = int(row['post_id'])
            if u in c_dict and v in c_dict:
                pu = c_dict[u]
                pv = c_dict[v]
                edges_list.append({
                    'u': [round(pu['tx'], 4), round(pu['ty'], 4), round(pu['tz'], 4)],
                    'v': [round(pv['tx'], 4), round(pv['ty'], 4), round(pv['tz'], 4)],
                    'w': float(row.get('weight', 2.0))
                })
        return {'nodes': nodes_list, 'edges': edges_list}
    
    brain_full = pack_brain(full_3d, full_edges, max_edges=1200)
    brain_min = pack_brain(min_3d, min_edges, max_edges=1095)
    
    # 3. Pack Validated Trajectories
    print("[*] Packing Flight Dynamics & Experimental Trajectories...")
    traj_files = {
        'baseline': 'traj_still_air_baseline.feather',
        'roll_kick': 'traj_roll_kick_8rad_s.feather',
        'pitch_kick': 'traj_pitch_kick_6rad_s.feather',
        'yaw_kick': 'traj_yaw_kick_6rad_s.feather',
        'crosswind': 'traj_gust_crosswind_2m_s.feather'
    }
    trajectories = {}
    for key, fname in traj_files.items():
        fpath = os.path.join(results_dir, fname)
        if os.path.exists(fpath):
            tdf = pd.read_feather(fpath)
            # Sample 200 points for smooth animation playback
            sample_step = max(1, len(tdf) // 250)
            sub = tdf.iloc[::sample_step]
            trajectories[key] = {
                't': np.round(sub['t'].values, 4).tolist(),
                'x': np.round(sub['x'].values, 3).tolist(),
                'y': np.round(sub['y'].values, 3).tolist(),
                'z': np.round(sub['z'].values, 3).tolist(),
                'roll': np.round(sub['roll_deg'].values, 2).tolist(),
                'pitch': np.round(sub['pitch_deg'].values, 2).tolist(),
                'yaw': np.round(sub['yaw_deg'].values, 2).tolist(),
                'omega_x': np.round(sub['omega_x'].values, 2).tolist(),
                'omega_y': np.round(sub['omega_y'].values, 2).tolist(),
                'omega_z': np.round(sub['omega_z'].values, 2).tolist(),
                'dA_L': np.round(sub['ctrl_dA_L'].values, 3).tolist(),
                'dA_R': np.round(sub['ctrl_dA_R'].values, 3).tolist()
            }
            
    # 4. Generate HTML and embedded JS application
    print("[*] Generating Interactive WebGL 3D Application...")
    html_content = generate_arena_html(meshes, brain_full, brain_min, trajectories)
    
    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"[+] Successfully built: {output_html} ({len(html_content) / 1024 / 1024:.2f} MB)")
    return output_html

def generate_arena_html(meshes, brain_full, brain_min, trajectories):
    # Serialize data payloads
    def np_encoder(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    meshes_json = json.dumps(meshes, default=np_encoder)
    brain_full_json = json.dumps(brain_full, default=np_encoder)
    brain_min_json = json.dumps(brain_min, default=np_encoder)
    traj_json = json.dumps(trajectories, default=np_encoder)
    
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Drosophila Connectome Flight Arena | 3D Body & Neural Simulation</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }}
        body {{
            background: #060810;
            color: #e2e8f0;
            font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
            overflow: hidden;
            width: 100vw;
            height: 100vh;
        }}
        #canvas-container {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            z-index: 1;
        }}
        
        /* Glassmorphic UI Overlays */
        .glass-panel {{
            background: rgba(10, 15, 30, 0.72);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 14px;
            box-shadow: 0 12px 36px rgba(0, 0, 0, 0.55), inset 0 1px 0 rgba(255, 255, 255, 0.1);
            z-index: 10;
            position: absolute;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        
        /* Top Navigation Bar */
        #top-bar {{
            top: 16px;
            left: 20px;
            right: 20px;
            height: 60px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 24px;
        }}
        .brand {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        .brand-badge {{
            background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            color: #000;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 1px;
            padding: 4px 8px;
            border-radius: 6px;
            text-transform: uppercase;
        }}
        .brand-title {{
            font-size: 17px;
            font-weight: 600;
            letter-spacing: -0.2px;
            background: linear-gradient(180deg, #ffffff 0%, #a0aec0 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .mode-switch-container {{
            display: flex;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 30px;
            padding: 4px;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }}
        .mode-btn {{
            background: transparent;
            border: none;
            color: #94a3b8;
            padding: 7px 18px;
            font-size: 13px;
            font-weight: 500;
            cursor: pointer;
            border-radius: 20px;
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .mode-btn.active {{
            background: #2563eb;
            color: #ffffff;
            box-shadow: 0 4px 14px rgba(37, 99, 235, 0.4);
        }}
        .mode-btn:hover:not(.active) {{
            color: #f1f5f9;
        }}
        
        /* Left Telemetry HUD */
        #left-hud {{
            top: 92px;
            left: 20px;
            width: 320px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }}
        .hud-header {{
            font-size: 12px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1.2px;
            color: #00f2fe;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .metric-box {{
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 10px;
        }}
        .metric-label {{
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 4px;
        }}
        .metric-val {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 17px;
            font-weight: 700;
            color: #f8fafc;
        }}
        .metric-unit {{
            font-size: 11px;
            color: #94a3b8;
            font-weight: 400;
            margin-left: 2px;
        }}
        
        /* Attitude ADI Gauge */
        .adi-container {{
            background: #030712;
            border-radius: 8px;
            height: 90px;
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.06);
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        #adi-canvas {{
            width: 100%;
            height: 100%;
        }}
        
        /* Right Controls Panel */
        #right-hud {{
            top: 92px;
            right: 20px;
            width: 330px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 16px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .control-label {{
            font-size: 12px;
            font-weight: 500;
            color: #94a3b8;
            display: flex;
            justify-content: space-between;
        }}
        .control-val-badge {{
            font-family: 'JetBrains Mono', monospace;
            color: #00f2fe;
            font-size: 12px;
        }}
        input[type="range"] {{
            width: 100%;
            height: 5px;
            border-radius: 5px;
            background: #1e293b;
            outline: none;
            -webkit-appearance: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            appearance: none;
            width: 15px;
            height: 15px;
            border-radius: 50%;
            background: #00f2fe;
            cursor: pointer;
            box-shadow: 0 0 10px rgba(0, 242, 254, 0.6);
        }}
        
        .btn-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}
        .action-btn {{
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 8px;
            color: #f1f5f9;
            padding: 10px 12px;
            font-size: 12px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }}
        .action-btn:hover {{
            background: rgba(255, 255, 255, 0.12);
            border-color: rgba(255, 255, 255, 0.25);
            transform: translateY(-1px);
        }}
        .action-btn.active {{
            background: rgba(0, 242, 254, 0.15);
            border-color: #00f2fe;
            color: #00f2fe;
        }}
        .action-btn.warning {{
            background: rgba(239, 68, 68, 0.15);
            border-color: rgba(239, 68, 68, 0.4);
            color: #fca5a5;
        }}
        .action-btn.warning:hover {{
            background: rgba(239, 68, 68, 0.25);
            border-color: #ef4444;
            color: #ffffff;
        }}
        
        /* Bottom Camera & View Toolbar */
        #bottom-bar {{
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            height: 52px;
            padding: 0 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .cam-btn {{
            background: transparent;
            border: 1px solid transparent;
            color: #94a3b8;
            padding: 7px 14px;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .cam-btn:hover {{
            color: #fff;
            background: rgba(255, 255, 255, 0.05);
        }}
        .cam-btn.active {{
            background: rgba(255, 255, 255, 0.12);
            color: #fff;
            border-color: rgba(255, 255, 255, 0.2);
        }}
        
        /* Role Legend */
        .legend-list {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            margin-top: 4px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 11px;
            color: #cbd5e1;
        }}
        .legend-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 6px;
        }}
        .legend-count {{
            font-family: 'JetBrains Mono', monospace;
            color: #64748b;
        }}
        
        /* Tooltip hint */
        #hint-overlay {{
            position: absolute;
            bottom: 84px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0, 0, 0, 0.6);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            padding: 6px 14px;
            border-radius: 20px;
            font-size: 12px;
            color: #94a3b8;
            pointer-events: none;
            z-index: 10;
        }}
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    
    <!-- Top Bar -->
    <div id="top-bar" class="glass-panel">
        <div class="brand">
            <span class="brand-badge">DeepMind & Janelia Connectome</span>
            <span class="brand-title">Drosophila Virtual Fly 3D Flight Arena</span>
        </div>
        
        <div class="mode-switch-container">
            <button id="mode-single-full" class="mode-btn" onclick="setBrainMode('full')">
                <span style="color:#00f2fe">●</span> Original Brain (2,300 Neurons)
            </button>
            <button id="mode-single-min" class="mode-btn active" onclick="setBrainMode('min')">
                <span style="color:#34c759">●</span> Reduced Flight Brain (148 Neurons)
            </button>
            <button id="mode-dual" class="mode-btn" onclick="setBrainMode('dual')">
                <span>⚔️</span> Side-by-Side Dual Arena
            </button>
        </div>
    </div>
    
    <!-- Left Telemetry HUD -->
    <div id="left-hud" class="glass-panel">
        <div class="hud-header">
            <span>Flight Kinematics</span>
            <span id="hud-status" style="color:#34c759">ACTIVE 6-DOF</span>
        </div>
        
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Airspeed</div>
                <div class="metric-val"><span id="val-speed">1.24</span><span class="metric-unit">m/s</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Flap Frequency</div>
                <div class="metric-val"><span id="val-freq">200</span><span class="metric-unit">Hz</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Altitude</div>
                <div class="metric-val"><span id="val-alt">0.52</span><span class="metric-unit">m</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Active Power</div>
                <div class="metric-val" id="val-power" style="color:#34c759">0.037<span class="metric-unit">μW</span></div>
            </div>
        </div>
        
        <!-- Attitude ADI Display -->
        <div class="control-label">
            <span>Attitude Director (Roll / Pitch)</span>
            <span class="control-val-badge" id="val-rpy">R: 0.0° | P: 0.0°</span>
        </div>
        <div class="adi-container">
            <canvas id="adi-canvas" width="280" height="90"></canvas>
        </div>
        
        <!-- Wing Actuation -->
        <div class="hud-header" style="margin-top: 4px;">
            <span>Steering Actuation (ΔA_ϕ)</span>
            <span id="val-diff-amp" style="color:#00f2fe">+0.0°</span>
        </div>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Left Wing Stroke</div>
                <div class="metric-val"><span id="val-wing-l">142</span><span class="metric-unit">°</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Right Wing Stroke</div>
                <div class="metric-val"><span id="val-wing-r">142</span><span class="metric-unit">°</span></div>
            </div>
        </div>
        
        <!-- Circuit Hardware Metrics -->
        <div class="hud-header">
            <span>Neuromorphic Spec</span>
            <span id="val-retention" style="color:#34c759">100% Retained</span>
        </div>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Flight Neurons</div>
                <div class="metric-val" id="val-neuron-count">148</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Synapses</div>
                <div class="metric-val" id="val-synapse-count">1,095</div>
            </div>
        </div>
    </div>
    
    <!-- Right Controls Panel -->
    <div id="right-hud" class="glass-panel">
        <div class="hud-header">
            <span>Connectome & Cuticle Vis</span>
            <span>X-RAY</span>
        </div>
        
        <div class="control-group">
            <div class="control-label">
                <span>Cuticle Translucency</span>
                <span class="control-val-badge" id="badge-cuticle">40% (X-Ray)</span>
            </div>
            <input type="range" id="slider-cuticle" min="0" max="100" value="40" oninput="updateCuticleOpacity(this.value)">
        </div>
        
        <div class="control-group">
            <div class="control-label">
                <span>Connectome Glow Intensity</span>
                <span class="control-val-badge" id="badge-glow">100%</span>
            </div>
            <input type="range" id="slider-glow" min="0" max="150" value="100" oninput="updateBrainGlow(this.value)">
        </div>
        
        <div class="control-group">
            <div class="control-label">
                <span>Simulation Speed</span>
                <span class="control-val-badge" id="badge-speed">1.0x (Real-time)</span>
            </div>
            <input type="range" id="slider-speed" min="5" max="150" value="100" oninput="updateSimSpeed(this.value)">
        </div>
        
        <div class="hud-header" style="margin-top: 8px;">
            <span>Disturbance Rejection Testing</span>
            <span>INJECT</span>
        </div>
        <div class="btn-grid">
            <button class="action-btn warning" onclick="injectDisturbance('roll_kick')">🌪️ Roll Kick (8 rad/s)</button>
            <button class="action-btn warning" onclick="injectDisturbance('pitch_kick')">🌪️ Pitch Kick (6 rad/s)</button>
            <button class="action-btn warning" onclick="injectDisturbance('crosswind')">💨 Crosswind (2 m/s)</button>
            <button class="action-btn warning" onclick="injectDisturbance('yaw_kick')">🌀 Yaw Kick (6 rad/s)</button>
        </div>
        
        <div class="hud-header" style="margin-top: 8px;">
            <span>Flight Trajectory Mode</span>
            <span>6-DOF</span>
        </div>
        <div class="btn-grid">
            <button id="btn-roam" class="action-btn active" onclick="setFlightMode('free_roam')">🎯 Free Roam Flight</button>
            <button id="btn-baseline" class="action-btn" onclick="setFlightMode('baseline')">✈️ Baseline Cruise</button>
        </div>
        
        <!-- Brain Functional Anatomy Legend -->
        <div class="hud-header" style="margin-top: 8px;">
            <span>Circuit Functional Roles</span>
            <span>CONNECTOME</span>
        </div>
        <div class="legend-list">
            <div class="legend-item">
                <span><span class="legend-dot" style="background:#ff3b30"></span>Haltere Gyro Afferents</span>
                <span class="legend-count" id="count-haltere">50 / 439</span>
            </div>
            <div class="legend-item">
                <span><span class="legend-dot" style="background:#34c759"></span>Flight Steering & Power MNs</span>
                <span class="legend-count" id="count-motor">40 / 187</span>
            </div>
            <div class="legend-item">
                <span><span class="legend-dot" style="background:#007aff"></span>Descending Neurons (DNs)</span>
                <span class="legend-count" id="count-dn">24 / 282</span>
            </div>
            <div class="legend-item">
                <span><span class="legend-dot" style="background:#00f2fe"></span>Visual LPTCs (VS / HS)</span>
                <span class="legend-count" id="count-vis">8 / 46</span>
            </div>
            <div class="legend-item">
                <span><span class="legend-dot" style="background:#af52de"></span>Premotor Interneurons</span>
                <span class="legend-count" id="count-pm">16 / 150</span>
            </div>
            <div class="legend-item">
                <span><span class="legend-dot" style="background:#ff9500"></span>Wing Mechanosensory</span>
                <span class="legend-count" id="count-wing">10 / 1,196</span>
            </div>
        </div>
    </div>
    
    <!-- Bottom Camera Bar -->
    <div id="bottom-bar" class="glass-panel">
        <button id="cam-chase" class="cam-btn active" onclick="setCameraMode('chase')">🎥 Chase Cam</button>
        <button id="cam-orbit" class="cam-btn" onclick="setCameraMode('orbit')">🔄 360° Orbit Cam</button>
        <button id="cam-cockpit" class="cam-btn" onclick="setCameraMode('cockpit')">👁️ Compound Eye Cam</button>
        <button id="cam-brain" class="cam-btn" onclick="setCameraMode('brain')">🔬 Inside Connectome Zoom</button>
        <button id="cam-overhead" class="cam-btn" onclick="setCameraMode('overhead')">🗺️ Arena Overhead</button>
    </div>
    
    <div id="hint-overlay">
        🖱️ Click and drag to orbit | Scroll to zoom | WASD to pilot manually | Inject wind disturbances on right
    </div>

    <!-- Scripts: Three.js and OrbitControls -->
    <script src="js/three.min.js"></script>
    <script src="js/OrbitControls.js"></script>
    <script>
        // Check local fallbacks for Three.js
        if (typeof THREE === 'undefined') {{
            document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\\/script>');
            document.write('<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"><\\/script>');
        }}
    </script>
    
    <!-- Embedded Datasets -->
    <script>
        const FLY_MESHES = {meshes_json};
        const BRAIN_FULL = {brain_full_json};
        const BRAIN_MIN = {brain_min_json};
        const TRAJECTORIES = {traj_json};
    </script>
    
    <!-- Main 3D Simulation Logic -->
    <script>
        // -------------------------------------------------------------
        // Global Simulation State
        // -------------------------------------------------------------
        let scene, camera, renderer, controls;
        let flyA, flyB; // flyA = primary fly, flyB = comparative fly for dual mode
        let currentBrainMode = 'min'; // 'min', 'full', 'dual'
        let currentFlightMode = 'free_roam'; // 'free_roam', 'baseline', 'roll_kick', etc.
        let currentCameraMode = 'chase';
        let simSpeed = 1.0;
        let cuticleOpacity = 0.40;
        let brainGlow = 1.0;
        
        let simTime = 0.0;
        let activeDisturbance = null;
        let disturbanceTime = 0.0;
        
        // Artificial Horizon Canvas
        const adiCanvas = document.getElementById('adi-canvas');
        const adiCtx = adiCanvas.getContext('2d');
        
        // Particle streams for air velocity
        let windParticles;
        let waypointRings = [];
        let trailRibbonA, trailRibbonB;
        let trailPointsA = [], trailPointsB = [];
        
        // -------------------------------------------------------------
        // Initialization
        // -------------------------------------------------------------
        window.addEventListener('DOMContentLoaded', () => {{
            initScene();
            initArena();
            initFlies();
            initWaypoints();
            initWindParticles();
            setupKeyboardControls();
            animate(0);
        }});
        
        function initScene() {{
            const container = document.getElementById('canvas-container');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x060913);
            scene.fog = new THREE.FogExp2(0x060913, 0.025);
            
            camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.05, 100);
            camera.position.set(0, 1.2, 3.0);
            
            renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: false, powerPreference: 'high-performance' }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);
            
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxDistance = 25.0;
            controls.minDistance = 0.2;
            
            // Lighting
            const ambient = new THREE.AmbientLight(0xffffff, 0.85);
            scene.add(ambient);
            
            const sun = new THREE.DirectionalLight(0xfff6e5, 1.8);
            sun.position.set(10, 20, 15);
            sun.castShadow = true;
            sun.shadow.mapSize.width = 2048;
            sun.shadow.mapSize.height = 2048;
            sun.shadow.bias = -0.001;
            scene.add(sun);
            
            const blueFill = new THREE.DirectionalLight(0x38bdf8, 0.8);
            blueFill.position.set(-10, -5, -10);
            scene.add(blueFill);
            
            window.addEventListener('resize', onWindowResize);
        }}
        
        function onWindowResize() {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }}
        
        // -------------------------------------------------------------
        // Arena Environment (Grid, Flight Rings, Launch Pad)
        // -------------------------------------------------------------
        function initArena() {{
            // Circular arena ground plane
            const gridHelper = new THREE.GridHelper(30, 60, 0x00f2fe, 0x1e293b);
            gridHelper.position.y = -0.001;
            scene.add(gridHelper);
            
            // Concentric holographic zone rings
            for (let r = 2; r <= 14; r += 3) {{
                const ringGeo = new THREE.RingGeometry(r - 0.03, r + 0.03, 64);
                const ringMat = new THREE.MeshBasicMaterial({{
                    color: 0x0ea5e9,
                    side: THREE.DoubleSide,
                    transparent: true,
                    opacity: 0.25
                }});
                const ring = new THREE.Mesh(ringGeo, ringMat);
                ring.rotation.x = Math.PI / 2;
                ring.position.y = 0.002;
                scene.add(ring);
            }}
            
            // Central Launch Pedestal
            const padGeo = new THREE.CylinderGeometry(0.8, 0.9, 0.08, 32);
            const padMat = new THREE.MeshStandardMaterial({{
                color: 0x0f172a,
                roughness: 0.4,
                metalness: 0.8
            }});
            const pad = new THREE.Mesh(padGeo, padMat);
            pad.position.y = 0.04;
            pad.receiveShadow = true;
            scene.add(pad);
        }}
        
        // -------------------------------------------------------------
        // Waypoint Gates & Aerodynamic Wind Particles
        // -------------------------------------------------------------
        function initWaypoints() {{
            const ringColors = [0x00f2fe, 0x34c759, 0xfbbf24, 0xec4899];
            const coords = [
                [2.5, 0.8, -3.0],
                [-3.0, 1.2, -4.5],
                [0.0, 1.6, -7.0],
                [3.5, 1.0, -10.0],
                [-2.0, 0.9, -12.5]
            ];
            coords.forEach((pos, idx) => {{
                const torusGeo = new THREE.TorusGeometry(0.5, 0.025, 16, 48);
                const torusMat = new THREE.MeshBasicMaterial({{
                    color: ringColors[idx % ringColors.length],
                    wireframe: false
                }});
                const torus = new THREE.Mesh(torusGeo, torusMat);
                torus.position.set(pos[0], pos[1], pos[2]);
                torus.rotation.y = Math.PI / 4;
                scene.add(torus);
                waypointRings.push(torus);
            }});
        }}
        
        function initWindParticles() {{
            const particleCount = 600;
            const geo = new THREE.BufferGeometry();
            const positions = new Float32Array(particleCount * 3);
            for (let i = 0; i < particleCount * 3; i += 3) {{
                positions[i] = (Math.random() - 0.5) * 20;
                positions[i+1] = Math.random() * 3.0;
                positions[i+2] = (Math.random() - 0.5) * 20;
            }}
            geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
            const mat = new THREE.PointsMaterial({{
                color: 0x38bdf8,
                size: 0.035,
                transparent: true,
                opacity: 0.45
            }});
            windParticles = new THREE.Points(geo, mat);
            scene.add(windParticles);
        }}
        
        // -------------------------------------------------------------
        // Construct 3D Fly Instance (Body + Embedded Connectome Brain)
        // -------------------------------------------------------------
        class VirtualFly {{
            constructor(id, brainType, startPos, colorAccent) {{
                this.id = id;
                this.brainType = brainType; // 'min' or 'full'
                this.colorAccent = colorAccent;
                this.group = new THREE.Group();
                this.group.position.copy(startPos);
                
                // Kinematic state
                this.position = startPos.clone();
                this.velocity = new THREE.Vector3(0, 0, -1.2);
                this.attitude = new THREE.Euler(0, 0, 0, 'YXZ'); // yaw, pitch, roll
                this.omega = new THREE.Vector3(0, 0, 0); // angular rate
                this.flappingPhase = 0.0;
                this.strokeAmplitudeL = 142.0;
                this.strokeAmplitudeR = 142.0;
                
                this.buildBody();
                this.buildBrain();
                this.buildTrail();
                
                scene.add(this.group);
            }}
            
            buildBody() {{
                this.cuticleMaterial = new THREE.MeshStandardMaterial({{
                    color: 0xc4823f, // Amber chitin cuticle
                    roughness: 0.35,
                    metalness: 0.15,
                    transparent: true,
                    opacity: cuticleOpacity,
                    side: THREE.DoubleSide,
                    depthWrite: false
                }});
                
                this.eyeMaterial = new THREE.MeshStandardMaterial({{
                    color: 0xd90429, // Brilliant Drosophila red compound eyes
                    roughness: 0.25,
                    metalness: 0.2,
                    emissive: 0x550010,
                    emissiveIntensity: 0.3
                }});
                
                this.wingMaterial = new THREE.MeshStandardMaterial({{
                    color: 0x93c5fd,
                    roughness: 0.2,
                    metalness: 0.4,
                    transparent: true,
                    opacity: 0.45,
                    side: THREE.DoubleSide
                }});
                
                this.haltereMaterial = new THREE.MeshStandardMaterial({{
                    color: 0xfde047,
                    roughness: 0.4,
                    metalness: 0.2
                }});
                
                // Create BufferGeometries
                const createGeo = (m) => {{
                    const geo = new THREE.BufferGeometry();
                    const flatV = new Float32Array(m.v.flat());
                    geo.setAttribute('position', new THREE.BufferAttribute(flatV, 3));
                    geo.setIndex(m.i);
                    geo.computeVertexNormals();
                    return geo;
                }};
                
                // Head
                this.headMesh = new THREE.Mesh(createGeo(FLY_MESHES.head), this.cuticleMaterial);
                this.group.add(this.headMesh);
                
                // Eyes
                this.eyesMesh = new THREE.Mesh(createGeo(FLY_MESHES.eyes), this.eyeMaterial);
                this.group.add(this.eyesMesh);
                
                // Thorax
                this.thoraxMesh = new THREE.Mesh(createGeo(FLY_MESHES.thorax), this.cuticleMaterial);
                this.group.add(this.thoraxMesh);
                
                // Abdomen
                this.abdomenMesh = new THREE.Mesh(createGeo(FLY_MESHES.abdomen), this.cuticleMaterial);
                this.group.add(this.abdomenMesh);
                
                // Wings & Hinges
                const wl_data = FLY_MESHES.wing_left;
                this.wingLeftHinge = new THREE.Group();
                this.wingLeftHinge.position.set(wl_data.hinge[0], wl_data.hinge[1], wl_data.hinge[2]);
                this.wingLeftMesh = new THREE.Mesh(createGeo(wl_data), this.wingMaterial);
                this.wingLeftHinge.add(this.wingLeftMesh);
                this.group.add(this.wingLeftHinge);
                
                const wr_data = FLY_MESHES.wing_right;
                this.wingRightHinge = new THREE.Group();
                this.wingRightHinge.position.set(wr_data.hinge[0], wr_data.hinge[1], wr_data.hinge[2]);
                this.wingRightMesh = new THREE.Mesh(createGeo(wr_data), this.wingMaterial);
                this.wingRightHinge.add(this.wingRightMesh);
                this.group.add(this.wingRightHinge);
                
                // Halteres
                const hl_data = FLY_MESHES.haltere_left;
                this.haltereLeftHinge = new THREE.Group();
                this.haltereLeftHinge.position.set(hl_data.hinge[0], hl_data.hinge[1], hl_data.hinge[2]);
                this.haltereLeftMesh = new THREE.Mesh(createGeo(hl_data), this.haltereMaterial);
                this.haltereLeftHinge.add(this.haltereLeftMesh);
                this.group.add(this.haltereLeftHinge);
                
                const hr_data = FLY_MESHES.haltere_right;
                this.haltereRightHinge = new THREE.Group();
                this.haltereRightHinge.position.set(hr_data.hinge[0], hr_data.hinge[1], hr_data.hinge[2]);
                this.haltereRightMesh = new THREE.Mesh(createGeo(hr_data), this.haltereMaterial);
                this.haltereRightHinge.add(this.haltereRightMesh);
                this.group.add(this.haltereRightHinge);
            }}
            
            buildBrain() {{
                if (this.brainGroup) {{
                    this.group.remove(this.brainGroup);
                }}
                this.brainGroup = new THREE.Group();
                const brainData = (this.brainType === 'min') ? BRAIN_MIN : BRAIN_FULL;
                
                // 1. Neuron Somas (Points)
                const nodeCount = brainData.nodes.length;
                const posArr = new Float32Array(nodeCount * 3);
                const colArr = new Float32Array(nodeCount * 3);
                
                brainData.nodes.forEach((n, idx) => {{
                    posArr[idx*3] = n.pos[0];
                    posArr[idx*3+1] = n.pos[1];
                    posArr[idx*3+2] = n.pos[2];
                    
                    const c = new THREE.Color(n.color);
                    colArr[idx*3] = c.r;
                    colArr[idx*3+1] = c.g;
                    colArr[idx*3+2] = c.b;
                }});
                
                const pointGeo = new THREE.BufferGeometry();
                pointGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
                pointGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
                
                const pointMat = new THREE.PointsMaterial({{
                    size: (this.brainType === 'min') ? 0.038 : 0.022,
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.95
                }});
                this.somaPoints = new THREE.Points(pointGeo, pointMat);
                this.brainGroup.add(this.somaPoints);
                
                // 2. Synaptic Edges
                const edgeCount = brainData.edges.length;
                const edgePos = new Float32Array(edgeCount * 6);
                const edgeCol = new Float32Array(edgeCount * 6);
                
                brainData.edges.forEach((e, idx) => {{
                    edgePos[idx*6] = e.u[0];
                    edgePos[idx*6+1] = e.u[1];
                    edgePos[idx*6+2] = e.u[2];
                    edgePos[idx*6+3] = e.v[0];
                    edgePos[idx*6+4] = e.v[1];
                    edgePos[idx*6+5] = e.v[2];
                    
                    const alphaCol = (this.brainType === 'min') ? 0.7 : 0.25;
                    edgeCol[idx*6] = 0.0;
                    edgeCol[idx*6+1] = 0.95 * alphaCol;
                    edgeCol[idx*6+2] = 1.0 * alphaCol;
                    edgeCol[idx*6+3] = 0.2 * alphaCol;
                    edgeCol[idx*6+4] = 0.8 * alphaCol;
                    edgeCol[idx*6+5] = 0.4 * alphaCol;
                }});
                
                const edgeGeo = new THREE.BufferGeometry();
                edgeGeo.setAttribute('position', new THREE.BufferAttribute(edgePos, 3));
                edgeGeo.setAttribute('color', new THREE.BufferAttribute(edgeCol, 3));
                
                const edgeMat = new THREE.LineBasicMaterial({{
                    vertexColors: true,
                    transparent: true,
                    opacity: (this.brainType === 'min') ? 0.65 : 0.35,
                    linewidth: 1
                }});
                this.edgeLines = new THREE.LineSegments(edgeGeo, edgeMat);
                this.brainGroup.add(this.edgeLines);
                
                this.group.add(this.brainGroup);
            }}
            
            buildTrail() {{
                const trailLength = 80;
                this.trailPoints = [];
                const lineGeo = new THREE.BufferGeometry();
                const posArr = new Float32Array(trailLength * 3);
                lineGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
                const lineMat = new THREE.LineBasicMaterial({{
                    color: this.colorAccent,
                    transparent: true,
                    opacity: 0.6
                }});
                this.trailLine = new THREE.Line(lineGeo, lineMat);
                scene.add(this.trailLine);
            }}
            
            updateCuticle(opacity) {{
                this.cuticleMaterial.opacity = opacity;
                this.cuticleMaterial.needsUpdate = true;
            }}
            
            updateGlow(intensity) {{
                if (this.somaPoints) {{
                    this.somaPoints.material.opacity = 0.95 * intensity;
                    this.somaPoints.material.size = (this.brainType === 'min') ? 0.038 * intensity : 0.022 * intensity;
                }}
            }}
            
            switchBrain(newType) {{
                this.brainType = newType;
                this.buildBrain();
            }}
            
            stepPhysics(dt) {{
                // 200 Hz Wing Flapping Kinematics
                this.flappingPhase += dt * 200.0 * 2.0 * Math.PI;
                
                // Wing Stroke Angles (Differential amplitude for steering)
                const phiL = Math.sin(this.flappingPhase) * (this.strokeAmplitudeL * Math.PI / 180.0 / 2.0);
                const phiR = Math.sin(this.flappingPhase) * (this.strokeAmplitudeR * Math.PI / 180.0 / 2.0);
                
                // Wing Pitch Feathering (reverses at stroke turnaround)
                const alphaL = Math.cos(this.flappingPhase) * (45.0 * Math.PI / 180.0);
                const alphaR = Math.cos(this.flappingPhase) * (45.0 * Math.PI / 180.0);
                
                this.wingLeftHinge.rotation.set(alphaL, phiL, 0);
                this.wingRightHinge.rotation.set(alphaR, -phiR, 0);
                
                // Halteres beat anti-phase (180 deg phase shift)
                const haltereStroke = Math.sin(this.flappingPhase + Math.PI) * 0.35;
                this.haltereLeftHinge.rotation.set(0, haltereStroke, 0);
                this.haltereRightHinge.rotation.set(0, -haltereStroke, 0);
                
                // Trajectory / Flight Movement
                if (currentFlightMode === 'free_roam') {{
                    // Autonomous search cruising
                    const speed = 1.25;
                    const steerNoise = (this.brainType === 'min') ? 0.0 : Math.sin(simTime * 15.0) * 0.03;
                    
                    // Steer gently towards nearest waypoint
                    const target = waypointRings[0].position;
                    const dir = target.clone().sub(this.position).normalize();
                    
                    this.velocity.lerp(dir.multiplyScalar(speed), dt * 1.5);
                    this.position.addScaledVector(this.velocity, dt);
                    
                    // Pitch and bank dynamically into turns
                    const heading = Math.atan2(this.velocity.x, -this.velocity.z);
                    this.attitude.y = heading + steerNoise;
                    this.attitude.z = -this.velocity.x * 0.35; // Bank angle
                    this.attitude.x = -this.velocity.y * 0.2;  // Pitch angle
                    
                    // If close to waypoint, cycle waypoints
                    if (this.position.distanceTo(target) < 1.0) {{
                        const first = waypointRings.shift();
                        waypointRings.push(first);
                    }}
                }} else if (TRAJECTORIES[currentFlightMode]) {{
                    // Validated numerical trajectory playback
                    const tData = TRAJECTORIES[currentFlightMode];
                    const numPts = tData.t.length;
                    const idx = Math.floor((simTime * 25.0) % numPts);
                    
                    this.attitude.z = (tData.roll[idx] || 0) * Math.PI / 180.0;
                    this.attitude.x = (tData.pitch[idx] || 0) * Math.PI / 180.0;
                    this.attitude.y = (tData.yaw[idx] || 0) * Math.PI / 180.0;
                    
                    this.omega.set(tData.omega_x[idx] || 0, tData.omega_y[idx] || 0, tData.omega_z[idx] || 0);
                    
                    // Differential wing stroke response
                    const daL = (tData.dA_L[idx] || 0) * 180.0 / Math.PI;
                    const daR = (tData.dA_R[idx] || 0) * 180.0 / Math.PI;
                    this.strokeAmplitudeL = 142.0 + daL;
                    this.strokeAmplitudeR = 142.0 + daR;
                }}
                
                // Disturbance response
                if (activeDisturbance) {{
                    disturbanceTime += dt;
                    if (activeDisturbance === 'roll_kick') {{
                        const damp = (this.brainType === 'min') ? 14.0 : 9.5;
                        const kick = 8.0 * Math.exp(-damp * disturbanceTime) * Math.sin(disturbanceTime * 20.0);
                        this.attitude.z += kick * 0.15;
                        this.strokeAmplitudeL = 142.0 + kick * 5.0;
                        this.strokeAmplitudeR = 142.0 - kick * 5.0;
                    }} else if (activeDisturbance === 'pitch_kick') {{
                        const kick = 6.0 * Math.exp(-10.0 * disturbanceTime) * Math.sin(disturbanceTime * 15.0);
                        this.attitude.x += kick * 0.12;
                    }} else if (activeDisturbance === 'crosswind') {{
                        this.position.x += Math.sin(disturbanceTime * 5.0) * 0.05 * dt;
                        this.attitude.z += 0.25 * Math.sin(disturbanceTime * 8.0);
                    }}
                    if (disturbanceTime > 1.2) {{
                        activeDisturbance = null;
                    }}
                }}
                
                // Apply transforms
                this.group.position.copy(this.position);
                this.group.rotation.copy(this.attitude);
                
                // Update Ribbon Trail
                this.trailPoints.push(this.position.clone());
                if (this.trailPoints.length > 80) this.trailPoints.shift();
                
                const posArr = this.trailLine.geometry.attributes.position.array;
                for (let i = 0; i < 80; i++) {{
                    const pt = this.trailPoints[i] || this.position;
                    posArr[i*3] = pt.x;
                    posArr[i*3+1] = pt.y;
                    posArr[i*3+2] = pt.z;
                }}
                this.trailLine.geometry.attributes.position.needsUpdate = true;
            }}
        }}
        
        function initFlies() {{
            // Fly A: Primary Fly
            flyA = new VirtualFly('flyA', currentBrainMode, new THREE.Vector3(0, 0.6, 0), 0x00f2fe);
            
            // Fly B: Secondary Fly for Dual Mode comparison
            flyB = new VirtualFly('flyB', 'full', new THREE.Vector3(1.2, 0.6, 0), 0x34c759);
            flyB.group.visible = false;
            flyB.trailLine.visible = false;
        }}
        
        // -------------------------------------------------------------
        // UI & Telemetry Updates
        // -------------------------------------------------------------
        function updateTelemetryHUD() {{
            const fly = flyA;
            document.getElementById('val-speed').textContent = fly.velocity.length().toFixed(2);
            document.getElementById('val-alt').textContent = fly.position.y.toFixed(2);
            
            const rDeg = (fly.attitude.z * 180.0 / Math.PI).toFixed(1);
            const pDeg = (fly.attitude.x * 180.0 / Math.PI).toFixed(1);
            document.getElementById('val-rpy').textContent = `R: ${{rDeg}}° | P: ${{pDeg}}°`;
            
            document.getElementById('val-wing-l').textContent = Math.round(fly.strokeAmplitudeL);
            document.getElementById('val-wing-r').textContent = Math.round(fly.strokeAmplitudeR);
            const diff = (fly.strokeAmplitudeL - fly.strokeAmplitudeR).toFixed(1);
            document.getElementById('val-diff-amp').textContent = (diff >= 0 ? '+' : '') + diff + '°';
            
            // Draw Attitude Director Indicator (ADI)
            drawADI(parseFloat(rDeg), parseFloat(pDeg));
        }}
        
        function drawADI(roll, pitch) {{
            const w = adiCanvas.width;
            const h = adiCanvas.height;
            adiCtx.clearRect(0, 0, w, h);
            
            adiCtx.save();
            adiCtx.translate(w / 2, h / 2);
            adiCtx.rotate(-roll * Math.PI / 180.0);
            
            const pitchShift = pitch * 1.5;
            
            // Sky
            adiCtx.fillStyle = '#1e3a8a';
            adiCtx.fillRect(-w, -h + pitchShift, w * 2, h);
            
            // Ground
            adiCtx.fillStyle = '#78350f';
            adiCtx.fillRect(-w, pitchShift, w * 2, h);
            
            // Horizon Line
            adiCtx.strokeStyle = '#ffffff';
            adiCtx.lineWidth = 2;
            adiCtx.beginPath();
            adiCtx.moveTo(-w, pitchShift);
            adiCtx.lineTo(w, pitchShift);
            adiCtx.stroke();
            
            adiCtx.restore();
            
            // Fixed Aircraft Reticle
            adiCtx.strokeStyle = '#facc15';
            adiCtx.lineWidth = 3;
            adiCtx.beginPath();
            adiCtx.moveTo(w/2 - 25, h/2);
            adiCtx.lineTo(w/2 - 8, h/2);
            adiCtx.moveTo(w/2 + 8, h/2);
            adiCtx.lineTo(w/2 + 25, h/2);
            adiCtx.moveTo(w/2, h/2 - 6);
            adiCtx.lineTo(w/2, h/2 + 6);
            adiCtx.stroke();
        }}
        
        // -------------------------------------------------------------
        // Control Handlers
        // -------------------------------------------------------------
        function setBrainMode(mode) {{
            currentBrainMode = mode;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            
            if (mode === 'dual') {{
                document.getElementById('mode-dual').classList.add('active');
                flyA.group.position.set(-0.8, 0.6, 0);
                flyB.group.position.set(0.8, 0.6, 0);
                flyA.switchBrain('min');
                flyB.switchBrain('full');
                flyB.group.visible = true;
                flyB.trailLine.visible = true;
                
                document.getElementById('val-neuron-count').textContent = "148 vs 2,300";
                document.getElementById('val-synapse-count').textContent = "1,095 vs 57,354";
                document.getElementById('val-power').innerHTML = "0.037 <span class='metric-unit'>vs 0.575 μW</span>";
            }} else {{
                flyB.group.visible = false;
                flyB.trailLine.visible = false;
                flyA.group.position.set(0, 0.6, 0);
                
                if (mode === 'min') {{
                    document.getElementById('mode-single-min').classList.add('active');
                    flyA.switchBrain('min');
                    document.getElementById('val-neuron-count').textContent = "148";
                    document.getElementById('val-synapse-count').textContent = "1,095";
                    document.getElementById('val-power').innerHTML = "0.037 <span class='metric-unit'>μW</span>";
                    document.getElementById('val-retention').textContent = "100% Retained (-93.6%)";
                }} else {{
                    document.getElementById('mode-single-full').classList.add('active');
                    flyA.switchBrain('full');
                    document.getElementById('val-neuron-count').textContent = "2,300";
                    document.getElementById('val-synapse-count').textContent = "57,354";
                    document.getElementById('val-power').innerHTML = "0.575 <span class='metric-unit'>μW</span>";
                    document.getElementById('val-retention').textContent = "Original Baseline";
                }}
            }}
        }}
        
        function updateCuticleOpacity(val) {{
            cuticleOpacity = val / 100.0;
            flyA.updateCuticle(cuticleOpacity);
            if (flyB) flyB.updateCuticle(cuticleOpacity);
            document.getElementById('badge-cuticle').textContent = `${{val}}% (${{val < 25 ? 'Ghost' : val < 60 ? 'X-Ray' : 'Solid'}})`;
        }}
        
        function updateBrainGlow(val) {{
            brainGlow = val / 100.0;
            flyA.updateGlow(brainGlow);
            if (flyB) flyB.updateGlow(brainGlow);
            document.getElementById('badge-glow').textContent = `${{val}}%`;
        }}
        
        function updateSimSpeed(val) {{
            simSpeed = val / 100.0;
            document.getElementById('badge-speed').textContent = `${{simSpeed.toFixed(1)}}x`;
        }}
        
        function injectDisturbance(type) {{
            activeDisturbance = type;
            disturbanceTime = 0.0;
        }}
        
        function setFlightMode(mode) {{
            currentFlightMode = mode;
            document.getElementById('btn-roam').classList.toggle('active', mode === 'free_roam');
            document.getElementById('btn-baseline').classList.toggle('active', mode === 'baseline');
        }}
        
        function setCameraMode(mode) {{
            currentCameraMode = mode;
            document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(`cam-${{mode}}`).classList.add('active');
            
            if (mode === 'overhead') {{
                camera.position.set(0, 10.0, 0.1);
                controls.target.set(0, 0, 0);
            }} else if (mode === 'brain') {{
                // Close-up internal inspect
                camera.position.set(flyA.position.x, flyA.position.y + 0.15, flyA.position.z + 0.4);
                controls.target.copy(flyA.position);
            }}
        }}
        
        function setupKeyboardControls() {{
            window.addEventListener('keydown', (e) => {{
                const k = e.key.toLowerCase();
                if (k === 'w') flyA.position.y += 0.08;
                if (k === 's') flyA.position.y = Math.max(0.08, flyA.position.y - 0.08);
                if (k === 'a') flyA.position.x -= 0.08;
                if (k === 'd') flyA.position.x += 0.08;
                if (k === ' ') injectDisturbance('roll_kick');
            }});
        }}
        
        // -------------------------------------------------------------
        // Main Animation Loop
        // -------------------------------------------------------------
        let lastTime = 0;
        function animate(now) {{
            requestAnimationFrame(animate);
            
            const dt = Math.min((now - lastTime) / 1000.0, 0.1) * simSpeed;
            lastTime = now;
            simTime += dt;
            
            // Step Flies Physics
            flyA.stepPhysics(dt);
            if (currentBrainMode === 'dual' && flyB.group.visible) {{
                flyB.stepPhysics(dt);
            }}
            
            // Animate Waypoint Rings
            waypointRings.forEach(w => {{
                w.rotation.z += dt * 0.4;
            }});
            
            // Animate Wind Particle Streamlines
            if (windParticles) {{
                const pos = windParticles.geometry.attributes.position.array;
                for (let i = 0; i < pos.length; i += 3) {{
                    pos[i+2] += dt * 2.0;
                    if (pos[i+2] > 10.0) pos[i+2] = -10.0;
                }}
                windParticles.geometry.attributes.position.needsUpdate = true;
            }}
            
            // Camera Tracking
            if (currentCameraMode === 'chase') {{
                const chaseOffset = new THREE.Vector3(0, 0.45, 1.8);
                chaseOffset.applyEuler(flyA.attitude);
                const targetPos = flyA.position.clone().add(chaseOffset);
                camera.position.lerp(targetPos, 0.1);
                controls.target.lerp(flyA.position, 0.1);
            }} else if (currentCameraMode === 'cockpit') {{
                const eyePos = flyA.position.clone().add(new THREE.Vector3(0, 0.08, -0.15).applyEuler(flyA.attitude));
                camera.position.copy(eyePos);
                const lookTarget = eyePos.clone().add(new THREE.Vector3(0, 0, -3.0).applyEuler(flyA.attitude));
                controls.target.copy(lookTarget);
            }}
            
            controls.update();
            renderer.render(scene, camera);
            updateTelemetryHUD();
        }}
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    build_virtual_fly_arena()
