"""
Builds the 3D Naturalistic Virtual Fly Environment matching the user's reference images:
- Sliced bananas/fruit on the arena floor
- Smooth river stones / obstacles
- Diffuse yellow odor plume wafting from food
- Distant rolling green hills and open sky
- Fruit fly with articulated walking legs, flight wings, and extendable feeding proboscis
- Embedded 3D Connectome Brain (Original 2,300-N vs Reduced 148-N)
"""

import os
import json
import numpy as np
import pandas as pd
from orthocop.visualization.build_virtual_fly_arena import (
    parse_obj_unique, to_threejs_coords, prepare_3d_coordinates
)

def build_nature_arena(
    output_html="visualizations/virtual_fly_nature_arena.html",
    asset_dir="visualizations/fly_assets",
    circuits_dir="circuits",
    results_dir="results"
):
    print("[*] Extracting 3D FlyBody Anatomical Meshes for Nature Arena...")
    os.makedirs(os.path.dirname(output_html), exist_ok=True)
    
    meshes = {}
    
    # Head
    head_v, head_i = parse_obj_unique(os.path.join(asset_dir, 'head_body.obj'))
    head_three = to_threejs_coords(head_v)
    meshes['head'] = {'v': np.round(head_three, 4).tolist(), 'i': head_i.tolist()}
    
    # Eyes
    eyes_v, eyes_i = parse_obj_unique(os.path.join(asset_dir, 'head_red.obj'), max_faces=12000)
    eyes_three = to_threejs_coords(eyes_v)
    meshes['eyes'] = {'v': np.round(eyes_three, 4).tolist(), 'i': eyes_i.tolist()}
    
    # Thorax
    thorax_v, thorax_i = parse_obj_unique(os.path.join(asset_dir, 'thorax_body.obj'))
    thorax_three = to_threejs_coords(thorax_v)
    meshes['thorax'] = {'v': np.round(thorax_three, 4).tolist(), 'i': thorax_i.tolist()}
    
    # Left & Right Wings (hinge-centered)
    l_hinge = np.array([0.0484, -0.4619, 1.3266])
    r_hinge = np.array([0.0484, 0.4619, 1.3266])
    
    wl_v, wl_i = parse_obj_unique(os.path.join(asset_dir, 'wing_left.obj'))
    wl_local = wl_v - l_hinge
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
    
    # Halteres
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
    
    # Abdomen segments combined
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
    
    # Connectome Brains
    print("[*] Processing Connectome Brains...")
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
        fly_x = -0.85 + (z_conn - 100.0) / 900.0 * 1.15
        fly_y = (x_conn - 390.0) / 200.0 * 0.28
        fly_z = 1.15 - (y_conn - 350.0) / 200.0 * 0.15
        verts = np.column_stack([fly_x, fly_y, fly_z])
        return to_threejs_coords(verts)
        
    full_pts = transform_brain_coords(full_3d)
    min_pts = transform_brain_coords(min_3d)
    
    full_3d['tx'], full_3d['ty'], full_3d['tz'] = full_pts[:, 0], full_pts[:, 1], full_pts[:, 2]
    min_3d['tx'], min_3d['ty'], min_3d['tz'] = min_pts[:, 0], min_pts[:, 1], min_pts[:, 2]
    
    role_colors = {
        'haltere_sensory': '#ff3b30',
        'wing_sensory': '#ff9500',
        'visual_sensory': '#00f2fe',
        'descending_neuron': '#007aff',
        'premotor_interneuron': '#af52de',
        'motor_neuron': '#34c759'
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
                'type': str(row.get('type', ''))
            })
        edges_list = []
        for _, row in edges_df.head(max_edges).iterrows():
            u, v = int(row['pre_id']), int(row['post_id'])
            if u in c_dict and v in c_dict:
                pu, pv = c_dict[u], c_dict[v]
                edges_list.append({
                    'u': [round(pu['tx'], 4), round(pu['ty'], 4), round(pu['tz'], 4)],
                    'v': [round(pv['tx'], 4), round(pv['ty'], 4), round(pv['tz'], 4)],
                    'w': float(row.get('weight', 2.0))
                })
        return {'nodes': nodes_list, 'edges': edges_list}
        
    brain_full = pack_brain(full_3d, full_edges, max_edges=1200)
    brain_min = pack_brain(min_3d, min_edges, max_edges=1095)
    
    html_content = generate_nature_arena_html(meshes, brain_full, brain_min)
    with open(output_html, 'w') as f:
        f.write(html_content)
    print(f"[+] Successfully generated Nature Arena: {output_html} ({len(html_content)/1024/1024:.2f} MB)")
    return output_html

def generate_nature_arena_html(meshes, brain_full, brain_min):
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

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FlyGym Nature Virtual Arena | Sliced Fruit, Odor Plume & Connectome Brain</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com">
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            user-select: none;
        }}
        body {{
            background: #e8ecef;
            color: #1e293b;
            font-family: 'Outfit', sans-serif;
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
        
        /* Modern Glass UI Overlays */
        .glass-panel {{
            background: rgba(255, 255, 255, 0.78);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.6);
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08), 0 1px 3px rgba(0, 0, 0, 0.05);
            z-index: 10;
            position: absolute;
            transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }}
        
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
            background: #f59e0b;
            color: #ffffff;
            font-size: 11px;
            font-weight: 700;
            letter-spacing: 0.8px;
            padding: 4px 8px;
            border-radius: 6px;
            text-transform: uppercase;
        }}
        .brand-title {{
            font-size: 17px;
            font-weight: 600;
            color: #0f172a;
        }}
        
        .mode-switch-container {{
            display: flex;
            background: rgba(0, 0, 0, 0.06);
            border-radius: 30px;
            padding: 4px;
        }}
        .mode-btn {{
            background: transparent;
            border: none;
            color: #64748b;
            padding: 7px 18px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            border-radius: 20px;
            transition: all 0.25s ease;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .mode-btn.active {{
            background: #ffffff;
            color: #0f172a;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.1);
        }}
        
        /* Left Telemetry */
        #left-hud {{
            top: 92px;
            left: 20px;
            width: 320px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }}
        .hud-header {{
            font-size: 12px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            color: #d97706;
            display: flex;
            justify-content: space-between;
            border-bottom: 1px solid rgba(0, 0, 0, 0.08);
            padding-bottom: 6px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .metric-box {{
            background: rgba(255, 255, 255, 0.85);
            border: 1px solid rgba(0, 0, 0, 0.05);
            border-radius: 8px;
            padding: 10px;
        }}
        .metric-label {{
            font-size: 11px;
            color: #64748b;
            text-transform: uppercase;
            font-weight: 600;
            margin-bottom: 4px;
        }}
        .metric-val {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 16px;
            font-weight: 700;
            color: #0f172a;
        }}
        .metric-unit {{
            font-size: 11px;
            color: #94a3b8;
            font-weight: 400;
            margin-left: 2px;
        }}
        
        /* Right Controls */
        #right-hud {{
            top: 92px;
            right: 20px;
            width: 330px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
            max-height: calc(100vh - 120px);
            overflow-y: auto;
        }}
        .control-group {{
            display: flex;
            flex-direction: column;
            gap: 6px;
        }}
        .control-label {{
            font-size: 12px;
            font-weight: 600;
            color: #475569;
            display: flex;
            justify-content: space-between;
        }}
        .control-val-badge {{
            font-family: 'JetBrains Mono', monospace;
            color: #d97706;
        }}
        input[type="range"] {{
            width: 100%;
            height: 6px;
            border-radius: 5px;
            background: #cbd5e1;
            outline: none;
            -webkit-appearance: none;
        }}
        input[type="range"]::-webkit-slider-thumb {{
            -webkit-appearance: none;
            width: 16px;
            height: 16px;
            border-radius: 50%;
            background: #f59e0b;
            cursor: pointer;
            box-shadow: 0 1px 6px rgba(245, 158, 11, 0.5);
        }}
        
        .btn-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}
        .action-btn {{
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(0, 0, 0, 0.1);
            border-radius: 8px;
            color: #1e293b;
            padding: 9px 12px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }}
        .action-btn:hover {{
            background: #ffffff;
            border-color: #f59e0b;
            transform: translateY(-1px);
            box-shadow: 0 4px 10px rgba(0,0,0,0.06);
        }}
        .action-btn.active {{
            background: #fef3c7;
            border-color: #f59e0b;
            color: #b45309;
        }}
        
        /* Bottom View Toolbar */
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
            border: none;
            color: #64748b;
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s ease;
            display: flex;
            align-items: center;
            gap: 6px;
        }}
        .cam-btn:hover {{
            color: #0f172a;
            background: rgba(0, 0, 0, 0.05);
        }}
        .cam-btn.active {{
            background: #ffffff;
            color: #0f172a;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
        }}
        
        #hint-overlay {{
            position: absolute;
            bottom: 84px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid rgba(0, 0, 0, 0.08);
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 12px;
            color: #475569;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            pointer-events: none;
            z-index: 10;
        }}
    </style>
</head>
<body>
    <div id="canvas-container"></div>
    
    <!-- Top Navigation -->
    <div id="top-bar" class="glass-panel">
        <div class="brand">
            <span class="brand-badge">FlyGym & Connectome</span>
            <span class="brand-title">Naturalistic Virtual Fruit Fly Arena</span>
        </div>
        
        <div class="mode-switch-container">
            <button id="mode-single-min" class="mode-btn active" onclick="setBrainMode('min')">
                <span style="color:#10b981">●</span> Reduced Flight Brain (148 Neurons)
            </button>
            <button id="mode-single-full" class="mode-btn" onclick="setBrainMode('full')">
                <span style="color:#0ea5e9">●</span> Original Brain (2,300 Neurons)
            </button>
            <button id="mode-dual" class="mode-btn" onclick="setBrainMode('dual')">
                <span>⚔️</span> Side-by-Side Dual Arena
            </button>
        </div>
    </div>
    
    <!-- Left Telemetry HUD -->
    <div id="left-hud" class="glass-panel">
        <div class="hud-header">
            <span>Locomotion & Behavior</span>
            <span id="val-state-badge" style="color:#10b981">WALKING TRIPOD</span>
        </div>
        
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-label">Locomotion Mode</div>
                <div class="metric-val" id="val-loco-mode" style="font-size:14px;">Ground Roam</div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Walking Speed</div>
                <div class="metric-val"><span id="val-speed">18.4</span><span class="metric-unit">mm/s</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Odor Intensity</div>
                <div class="metric-val"><span id="val-odor">0.82</span><span class="metric-unit">ppm</span></div>
            </div>
            <div class="metric-box">
                <div class="metric-label">Active Power</div>
                <div class="metric-val" id="val-power" style="color:#10b981">0.037<span class="metric-unit">μW</span></div>
            </div>
        </div>
        
        <div class="hud-header">
            <span>Connectome Brain Spec</span>
            <span id="val-retention" style="color:#10b981">100% Retained</span>
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
        
        <div class="hud-header">
            <span>Chemosensory Tracking</span>
            <span style="color:#f59e0b">BANANA PLUME</span>
        </div>
        <div style="font-size:12px; color:#64748b; line-height:1.5;">
            Antennal and maxillary palp olfactory receptors detect the acetate/butyrate plume emanating from sliced bananas, driving positive chemotaxis towards the food disk.
        </div>
    </div>
    
    <!-- Right Controls -->
    <div id="right-hud" class="glass-panel">
        <div class="hud-header">
            <span>Cuticle & Connectome Vis</span>
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
        
        <div class="hud-header" style="margin-top: 6px;">
            <span>Behavior Triggers</span>
            <span>ACTIONS</span>
        </div>
        <div class="btn-grid">
            <button id="btn-feed" class="action-btn active" onclick="setBehavior('feed')">🍌 Feed on Banana</button>
            <button id="btn-roam" class="action-btn" onclick="setBehavior('roam')">🐾 Roam & Search</button>
            <button id="btn-fly" class="action-btn" onclick="setBehavior('fly')">🛫 Take Off & Fly</button>
            <button id="btn-groom" class="action-btn" onclick="setBehavior('groom')">🧼 Head Grooming</button>
        </div>
        
        <div class="hud-header" style="margin-top: 6px;">
            <span>Environment Features</span>
            <span>TOGGLE</span>
        </div>
        <div class="btn-grid">
            <button id="btn-plume" class="action-btn active" onclick="toggleOdorPlume()">💨 Odor Plume</button>
            <button id="btn-stones" class="action-btn active" onclick="toggleStones()">🪨 River Stones</button>
        </div>
        
        <div class="hud-header" style="margin-top: 6px;">
            <span>Functional Anatomy</span>
            <span>CIRCUIT</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:6px; font-size:11px;">
            <div style="display:flex; justify-content:space-between;">
                <span><span style="color:#ff3b30">●</span> Haltere Gyro Units</span>
                <span style="font-family:monospace; color:#64748b;">50 / 439</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span><span style="color:#34c759">●</span> Steering & Power MNs</span>
                <span style="font-family:monospace; color:#64748b;">40 / 187</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span><span style="color:#007aff">●</span> Descending Tract (DNs)</span>
                <span style="font-family:monospace; color:#64748b;">24 / 282</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span><span style="color:#00f2fe">●</span> Visual LPTCs (Optic Flow)</span>
                <span style="font-family:monospace; color:#64748b;">8 / 46</span>
            </div>
            <div style="display:flex; justify-content:space-between;">
                <span><span style="color:#af52de">●</span> Premotor Interneurons</span>
                <span style="font-family:monospace; color:#64748b;">16 / 150</span>
            </div>
        </div>
    </div>
    
    <!-- Bottom Camera Modes matching user reference images -->
    <div id="bottom-bar" class="glass-panel">
        <button id="cam-feeder" class="cam-btn active" onclick="setCameraPreset('feeder')">🍌 Feeder Close-Up (Img 1, 5)</button>
        <button id="cam-portrait" class="cam-btn" onclick="setCameraPreset('portrait')">🌄 Hills Portrait (Img 2)</button>
        <button id="cam-plume" class="cam-btn" onclick="setCameraPreset('plume')">💨 Plume Overview (Img 3)</button>
        <button id="cam-topdown" class="cam-btn" onclick="setCameraPreset('topdown')">🔍 Top-Down Roam (Img 4)</button>
        <button id="cam-brain" class="cam-btn" onclick="setCameraPreset('brain')">🔬 Inside Brain X-Ray</button>
    </div>
    
    <div id="hint-overlay">
        🍌 Observe the fly feeding on sliced bananas, following odor plumes past stones, and flying with embedded connectome brains!
    </div>

    <!-- Scripts: Three.js and OrbitControls -->
    <script src="js/three.min.js"></script>
    <script src="js/OrbitControls.js"></script>
    <script>
        if (typeof THREE === 'undefined') {{
            document.write('<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"><\\/script>');
            document.write('<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"><\\/script>');
        }}
    </script>
    
    <script>
        const FLY_MESHES = {meshes_json};
        const BRAIN_FULL = {brain_full_json};
        const BRAIN_MIN = {brain_min_json};
    </script>
    
    <script>
        // -------------------------------------------------------------
        // Main Simulation Setup
        // -------------------------------------------------------------
        let scene, camera, renderer, controls;
        let flyA, flyB;
        let currentBrainMode = 'min';
        let currentBehavior = 'feed'; // 'feed', 'roam', 'fly', 'groom'
        let currentCamPreset = 'feeder';
        let cuticleOpacity = 0.40;
        let brainGlow = 1.0;
        let simTime = 0.0;
        
        let bananaSlices = [];
        let riverStones = [];
        let odorPlumeGroup;
        let backgroundHillsGroup;
        
        window.addEventListener('DOMContentLoaded', () => {{
            initScene();
            initNatureEnvironment();
            initFlies();
            setCameraPreset('feeder');
            animate(0);
        }});
        
        function initScene() {{
            const container = document.getElementById('canvas-container');
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0xdce7eb);
            scene.fog = new THREE.FogExp2(0xdce7eb, 0.012);
            
            camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.05, 200);
            
            renderer = new THREE.WebGLRenderer({{ antialias: true, powerPreference: 'high-performance' }});
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            renderer.shadowMap.enabled = true;
            renderer.shadowMap.type = THREE.PCFSoftShadowMap;
            container.appendChild(renderer.domElement);
            
            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxDistance = 30.0;
            controls.minDistance = 0.3;
            
            // Warm studio / natural sun lighting matching FlyGym screenshots
            const ambient = new THREE.AmbientLight(0xfff7ed, 0.9);
            scene.add(ambient);
            
            const sun = new THREE.DirectionalLight(0xffffff, 1.4);
            sun.position.set(6, 12, 8);
            sun.castShadow = true;
            sun.shadow.mapSize.width = 2048;
            sun.shadow.mapSize.height = 2048;
            sun.shadow.bias = -0.0005;
            scene.add(sun);
            
            const skyHemi = new THREE.HemisphereLight(0xdce7eb, 0x94a3b8, 0.6);
            scene.add(skyHemi);
            
            window.addEventListener('resize', () => {{
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            }});
        }}
        
        // -------------------------------------------------------------
        // Naturalistic Environment (Banana Slices, Stones, Plume, Hills)
        // -------------------------------------------------------------
        function initNatureEnvironment() {{
            // 1. Clean warm testing ground surface (as in Image 1, 3, 5)
            const floorGeo = new THREE.PlaneGeometry(80, 80);
            const floorMat = new THREE.MeshStandardMaterial({{
                color: 0xedebe6,
                roughness: 0.85,
                metalness: 0.05
            }});
            const floor = new THREE.Mesh(floorGeo, floorMat);
            floor.rotation.x = -Math.PI / 2;
            floor.position.y = 0.0;
            floor.receiveShadow = true;
            scene.add(floor);
            
            // Subtle circular arena border
            const ringGeo = new THREE.RingGeometry(8.0, 8.04, 64);
            const ringMat = new THREE.MeshBasicMaterial({{ color: 0xcfd8dc, side: THREE.DoubleSide }});
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.y = 0.001;
            scene.add(ring);
            
            // 2. Sliced Fruit (Bananas with yellow peel & cream pulp)
            const peelMat = new THREE.MeshStandardMaterial({{
                color: 0xfacc15, // Bright banana yellow peel
                roughness: 0.35,
                metalness: 0.1
            }});
            const pulpMat = new THREE.MeshStandardMaterial({{
                color: 0xfef9c3, // Cream-white pulp
                roughness: 0.45,
                metalness: 0.05
            }});
            
            // Create a compound banana slice mesh
            const createBananaSlice = (radius, height) => {{
                const group = new THREE.Group();
                // Outer peel cylinder
                const peelGeo = new THREE.CylinderGeometry(radius, radius, height, 32);
                const peelMesh = new THREE.Mesh(peelGeo, peelMat);
                peelMesh.castShadow = true;
                peelMesh.receiveShadow = true;
                group.add(peelMesh);
                
                // Top pulp disc
                const pulpGeo = new THREE.CircleGeometry(radius * 0.92, 32);
                const pulpTop = new THREE.Mesh(pulpGeo, pulpMat);
                pulpTop.rotation.x = -Math.PI / 2;
                pulpTop.position.y = height / 2 + 0.001;
                group.add(pulpTop);
                
                // Tiny center seed dot
                const dotGeo = new THREE.CircleGeometry(radius * 0.12, 16);
                const dotMat = new THREE.MeshBasicMaterial({{ color: 0x854d0e }});
                const dot = new THREE.Mesh(dotGeo, dotMat);
                dot.rotation.x = -Math.PI / 2;
                dot.position.y = height / 2 + 0.002;
                group.add(dot);
                
                return group;
            }};
            
            // Banana Slices pile (as in Image 1 & 5)
            const slicePositions = [
                // Feeding target slice right in front of the fly
                {{ x: 0.0, z: -0.65, y: 0.06, r: 0.42, h: 0.12, rotZ: 0.0 }},
                // Stack of slices to the left (Image 1)
                {{ x: -1.2, z: -0.7, y: 0.06, r: 0.45, h: 0.12, rotZ: 0.05 }},
                {{ x: -1.5, z: -0.9, y: 0.06, r: 0.46, h: 0.12, rotZ: -0.04 }},
                {{ x: -1.0, z: -1.1, y: 0.06, r: 0.44, h: 0.12, rotZ: 0.02 }},
                {{ x: -1.3, z: -0.85, y: 0.18, r: 0.44, h: 0.12, rotZ: 0.12 }}, // stacked on top
                {{ x: -1.8, z: -0.6, y: 0.06, r: 0.42, h: 0.12, rotZ: -0.15 }}
            ];
            
            slicePositions.forEach(p => {{
                const s = createBananaSlice(p.r, p.h);
                s.position.set(p.x, p.y, p.z);
                s.rotation.z = p.rotZ;
                scene.add(s);
                bananaSlices.push(s);
            }});
            
            // 3. Smooth River Stones / Pebbles (as in Image 3)
            const stoneMat = new THREE.MeshStandardMaterial({{
                color: 0x78716c, // Smooth warm slate grey
                roughness: 0.35,
                metalness: 0.1
            }});
            const stonePositions = [
                {{ x: 1.8, z: -2.2, rx: 0.35, ry: 0.18, rz: 0.45, rotY: 0.4 }},
                {{ x: 2.2, z: -1.4, rx: 0.28, ry: 0.14, rz: 0.38, rotY: -0.6 }},
                {{ x: 1.5, z: -3.0, rx: 0.32, ry: 0.16, rz: 0.42, rotY: 0.2 }}
            ];
            stonePositions.forEach(p => {{
                const geo = new THREE.SphereGeometry(1.0, 32, 16);
                geo.scale(p.rx, p.ry, p.rz);
                const mesh = new THREE.Mesh(geo, stoneMat);
                mesh.position.set(p.x, p.ry, p.z);
                mesh.rotation.y = p.rotY;
                mesh.castShadow = true;
                mesh.receiveShadow = true;
                scene.add(mesh);
                riverStones.push(mesh);
            }});
            
            // 4. Volumetric Odor Plume Cloud (as in Image 3)
            odorPlumeGroup = new THREE.Group();
            const plumeCount = 400;
            const plumeGeo = new THREE.BufferGeometry();
            const plumePos = new Float32Array(plumeCount * 3);
            for (let i = 0; i < plumeCount * 3; i += 3) {{
                const t = Math.random();
                plumePos[i] = (Math.random() - 0.5) * (0.4 + t * 1.5);
                plumePos[i+1] = 0.05 + Math.random() * (0.2 + t * 0.4);
                plumePos[i+2] = -0.65 + t * 3.5;
            }}
            plumeGeo.setAttribute('position', new THREE.BufferAttribute(plumePos, 3));
            const plumeMat = new THREE.PointsMaterial({{
                color: 0xfacc15,
                size: 0.08,
                transparent: true,
                opacity: 0.35
            }});
            const plumePoints = new THREE.Points(plumeGeo, plumeMat);
            odorPlumeGroup.add(plumePoints);
            scene.add(odorPlumeGroup);
            
            // 5. Rolling Green Hills Backdrop (as in Image 2)
            backgroundHillsGroup = new THREE.Group();
            const hillMat = new THREE.MeshStandardMaterial({{
                color: 0x15803d, // Lush green hillside
                roughness: 0.9,
                metalness: 0.0
            }});
            
            const hillConfigs = [
                {{ x: 0, z: -25, r: 18, h: 6.5, scaleZ: 0.6 }},
                {{ x: -20, z: -30, r: 22, h: 8.0, scaleZ: 0.7 }},
                {{ x: 22, z: -28, r: 20, h: 7.2, scaleZ: 0.65 }}
            ];
            hillConfigs.forEach(h => {{
                const geo = new THREE.SphereGeometry(h.r, 32, 16);
                geo.scale(1.0, h.h / h.r, h.scaleZ);
                const mesh = new THREE.Mesh(geo, hillMat);
                mesh.position.set(h.x, -h.r + h.h, h.z);
                backgroundHillsGroup.add(mesh);
            }});
            scene.add(backgroundHillsGroup);
        }}
        
        // -------------------------------------------------------------
        // Articulated Fly Class (Walking, Feeding Proboscis, Flight, Brain)
        // -------------------------------------------------------------
        class NatureFly {{
            constructor(id, brainType, startPos, colorAccent) {{
                this.id = id;
                this.brainType = brainType;
                this.colorAccent = colorAccent;
                this.group = new THREE.Group();
                this.group.position.copy(startPos);
                
                this.position = startPos.clone();
                this.attitude = new THREE.Euler(0, 0, 0, 'YXZ');
                this.walkPhase = 0.0;
                this.flappingPhase = 0.0;
                this.proboscisExtend = 0.0;
                
                this.buildBody();
                this.buildBrain();
                this.buildLegs();
                this.buildProboscis();
                
                scene.add(this.group);
            }}
            
            buildBody() {{
                this.cuticleMaterial = new THREE.MeshStandardMaterial({{
                    color: 0x5a3825,
                    roughness: 0.4,
                    metalness: 0.15,
                    transparent: true,
                    opacity: cuticleOpacity,
                    side: THREE.DoubleSide
                }});
                this.eyeMaterial = new THREE.MeshStandardMaterial({{
                    color: 0xd90429,
                    roughness: 0.25,
                    metalness: 0.2
                }});
                this.wingMaterial = new THREE.MeshStandardMaterial({{
                    color: 0xe2e8f0,
                    roughness: 0.2,
                    metalness: 0.3,
                    transparent: true,
                    opacity: 0.4,
                    side: THREE.DoubleSide
                }});
                this.haltereMaterial = new THREE.MeshStandardMaterial({{
                    color: 0xfacc15,
                    roughness: 0.4
                }});
                
                const createGeo = (m) => {{
                    const geo = new THREE.BufferGeometry();
                    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(m.v.flat()), 3));
                    geo.setIndex(m.i);
                    geo.computeVertexNormals();
                    return geo;
                }};
                
                this.headMesh = new THREE.Mesh(createGeo(FLY_MESHES.head), this.cuticleMaterial);
                this.group.add(this.headMesh);
                
                this.eyesMesh = new THREE.Mesh(createGeo(FLY_MESHES.eyes), this.eyeMaterial);
                this.group.add(this.eyesMesh);
                
                this.thoraxMesh = new THREE.Mesh(createGeo(FLY_MESHES.thorax), this.cuticleMaterial);
                this.group.add(this.thoraxMesh);
                
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
                this.haltereLeftHinge.add(new THREE.Mesh(createGeo(hl_data), this.haltereMaterial));
                this.group.add(this.haltereLeftHinge);
                
                const hr_data = FLY_MESHES.haltere_right;
                this.haltereRightHinge = new THREE.Group();
                this.haltereRightHinge.position.set(hr_data.hinge[0], hr_data.hinge[1], hr_data.hinge[2]);
                this.haltereRightHinge.add(new THREE.Mesh(createGeo(hr_data), this.haltereMaterial));
                this.group.add(this.haltereRightHinge);
            }}
            
            buildProboscis() {{
                this.proboscisGroup = new THREE.Group();
                this.proboscisGroup.position.set(0.0, -0.06, 0.85);
                const probMat = new THREE.MeshStandardMaterial({{ color: 0x854d0e, roughness: 0.5 }});
                
                const seg1Geo = new THREE.CylinderGeometry(0.035, 0.04, 0.14, 16);
                this.seg1 = new THREE.Mesh(seg1Geo, probMat);
                this.seg1.position.y = -0.07;
                this.proboscisGroup.add(this.seg1);
                
                this.seg2Group = new THREE.Group();
                this.seg2Group.position.y = -0.14;
                const seg2Geo = new THREE.CylinderGeometry(0.025, 0.035, 0.16, 16);
                const seg2 = new THREE.Mesh(seg2Geo, probMat);
                seg2.position.y = -0.08;
                this.seg2Group.add(seg2);
                
                const spongeGeo = new THREE.CylinderGeometry(0.05, 0.05, 0.03, 16);
                const spongeMat = new THREE.MeshStandardMaterial({{ color: 0xa16207 }});
                const sponge = new THREE.Mesh(spongeGeo, spongeMat);
                sponge.position.y = -0.16;
                this.seg2Group.add(sponge);
                
                this.proboscisGroup.add(this.seg2Group);
                this.group.add(this.proboscisGroup);
            }}
            
            buildLegs() {{
                this.legs = {{}};
                const legMat = new THREE.MeshStandardMaterial({{ color: 0x78716c, roughness: 0.5 }});
                const legConfigs = [
                    {{ id: 'LF', pos: [-0.18, -0.05, 0.25], dirX: -1, length: 0.45 }},
                    {{ id: 'LM', pos: [-0.22, -0.06, 0.00], dirX: -1, length: 0.50 }},
                    {{ id: 'LH', pos: [-0.18, -0.05, -0.25], dirX: -1, length: 0.55 }},
                    {{ id: 'RF', pos: [0.18, -0.05, 0.25], dirX: 1, length: 0.45 }},
                    {{ id: 'RM', pos: [0.22, -0.06, 0.00], dirX: 1, length: 0.50 }},
                    {{ id: 'RH', pos: [0.18, -0.05, -0.25], dirX: 1, length: 0.55 }}
                ];
                
                legConfigs.forEach(cfg => {{
                    const rootGroup = new THREE.Group();
                    rootGroup.position.set(cfg.pos[0], cfg.pos[1], cfg.pos[2]);
                    
                    const femurGeo = new THREE.CylinderGeometry(0.02, 0.025, cfg.length * 0.55, 12);
                    const femur = new THREE.Mesh(femurGeo, legMat);
                    femur.position.set(cfg.dirX * 0.08, -0.1, 0);
                    femur.rotation.z = cfg.dirX * 0.7;
                    rootGroup.add(femur);
                    
                    const tibiaGroup = new THREE.Group();
                    tibiaGroup.position.set(cfg.dirX * 0.16, -0.18, 0);
                    const tibiaGeo = new THREE.CylinderGeometry(0.015, 0.018, cfg.length * 0.65, 12);
                    const tibia = new THREE.Mesh(tibiaGeo, legMat);
                    tibia.position.set(cfg.dirX * 0.06, -0.14, 0);
                    tibia.rotation.z = -cfg.dirX * 0.4;
                    tibiaGroup.add(tibia);
                    
                    rootGroup.add(tibiaGroup);
                    this.group.add(rootGroup);
                    this.legs[cfg.id] = {{ root: rootGroup, tibia: tibiaGroup, cfg: cfg }};
                }});
            }}
            
            buildBrain() {{
                if (this.brainGroup) this.group.remove(this.brainGroup);
                this.brainGroup = new THREE.Group();
                const bData = (this.brainType === 'min') ? BRAIN_MIN : BRAIN_FULL;
                
                const count = bData.nodes.length;
                const posArr = new Float32Array(count * 3);
                const colArr = new Float32Array(count * 3);
                bData.nodes.forEach((n, i) => {{
                    posArr[i*3] = n.pos[0];
                    posArr[i*3+1] = n.pos[1];
                    posArr[i*3+2] = n.pos[2];
                    const c = new THREE.Color(n.color);
                    colArr[i*3] = c.r; colArr[i*3+1] = c.g; colArr[i*3+2] = c.b;
                }});
                const pGeo = new THREE.BufferGeometry();
                pGeo.setAttribute('position', new THREE.BufferAttribute(posArr, 3));
                pGeo.setAttribute('color', new THREE.BufferAttribute(colArr, 3));
                const pMat = new THREE.PointsMaterial({{
                    size: (this.brainType === 'min') ? 0.038 : 0.022,
                    vertexColors: true,
                    transparent: true,
                    opacity: 0.95
                }});
                this.somaPoints = new THREE.Points(pGeo, pMat);
                this.brainGroup.add(this.somaPoints);
                
                const eCount = bData.edges.length;
                const ePos = new Float32Array(eCount * 6);
                const eCol = new Float32Array(eCount * 6);
                bData.edges.forEach((e, i) => {{
                    ePos[i*6] = e.u[0]; ePos[i*6+1] = e.u[1]; ePos[i*6+2] = e.u[2];
                    ePos[i*6+3] = e.v[0]; ePos[i*6+4] = e.v[1]; ePos[i*6+5] = e.v[2];
                    const a = (this.brainType === 'min') ? 0.7 : 0.25;
                    eCol[i*6] = 0; eCol[i*6+1] = a; eCol[i*6+2] = a;
                    eCol[i*6+3] = 0.2*a; eCol[i*6+4] = 0.8*a; eCol[i*6+5] = 0.4*a;
                }});
                const eGeo = new THREE.BufferGeometry();
                eGeo.setAttribute('position', new THREE.BufferAttribute(ePos, 3));
                eGeo.setAttribute('color', new THREE.BufferAttribute(eCol, 3));
                const eMat = new THREE.LineBasicMaterial({{
                    vertexColors: true,
                    transparent: true,
                    opacity: (this.brainType === 'min') ? 0.7 : 0.35
                }});
                this.edgeLines = new THREE.LineSegments(eGeo, eMat);
                this.brainGroup.add(this.edgeLines);
                
                this.group.add(this.brainGroup);
            }}
            
            updateCuticle(op) {{
                this.cuticleMaterial.opacity = op;
            }}
            
            updateGlow(intensity) {{
                if (this.somaPoints) {{
                    this.somaPoints.material.size = ((this.brainType === 'min') ? 0.038 : 0.022) * intensity;
                }}
            }}
            
            switchBrain(t) {{
                this.brainType = t;
                this.buildBrain();
            }}
            
            step(dt) {{
                if (currentBehavior === 'feed') {{
                    this.position.set(0.0, 0.16, -0.22);
                    this.attitude.set(-0.25, Math.PI, 0);
                    
                    this.proboscisExtend = Math.min(1.0, this.proboscisExtend + dt * 3.0);
                    this.seg2Group.position.y = -0.14 - this.proboscisExtend * 0.08;
                    this.seg2Group.rotation.x = this.proboscisExtend * 0.2;
                    
                    this.wingLeftHinge.rotation.set(0.1, -0.2, 0.05);
                    this.wingRightHinge.rotation.set(0.1, 0.2, -0.05);
                    
                    Object.values(this.legs).forEach(l => {{
                        l.root.rotation.x = 0.0;
                    }});
                }} else if (currentBehavior === 'roam') {{
                    this.proboscisExtend = Math.max(0.0, this.proboscisExtend - dt * 3.0);
                    this.seg2Group.position.y = -0.14 - this.proboscisExtend * 0.08;
                    
                    this.walkPhase += dt * 8.0;
                    const stepAmp = 0.25;
                    
                    const swing1 = Math.sin(this.walkPhase) * stepAmp;
                    const swing2 = Math.sin(this.walkPhase + Math.PI) * stepAmp;
                    
                    this.legs['LF'].root.rotation.x = swing1;
                    this.legs['RM'].root.rotation.x = -swing1;
                    this.legs['LH'].root.rotation.x = swing1;
                    
                    this.legs['RF'].root.rotation.x = swing2;
                    this.legs['LM'].root.rotation.x = -swing2;
                    this.legs['RH'].root.rotation.x = swing2;
                    
                    const radius = 1.4;
                    const walkAngle = simTime * 0.4;
                    this.position.x = Math.sin(walkAngle) * radius;
                    this.position.z = Math.cos(walkAngle) * radius - 0.5;
                    this.position.y = 0.16;
                    this.attitude.set(0, walkAngle + Math.PI / 2, 0);
                    
                    this.wingLeftHinge.rotation.set(0.08, -0.15, 0.05);
                    this.wingRightHinge.rotation.set(0.08, 0.15, -0.05);
                }} else if (currentBehavior === 'fly') {{
                    this.flappingPhase += dt * 200.0 * 2.0 * Math.PI;
                    this.proboscisExtend = 0.0;
                    
                    const strokeL = Math.sin(this.flappingPhase) * 1.1;
                    const strokeR = Math.sin(this.flappingPhase) * 1.1;
                    this.wingLeftHinge.rotation.set(0.3, strokeL, 0);
                    this.wingRightHinge.rotation.set(0.3, -strokeR, 0);
                    
                    const hStroke = Math.sin(this.flappingPhase + Math.PI) * 0.35;
                    this.haltereLeftHinge.rotation.set(0, hStroke, 0);
                    this.haltereRightHinge.rotation.set(0, -hStroke, 0);
                    
                    const flightRadius = 2.2;
                    const flightAngle = simTime * 0.8;
                    this.position.x = Math.sin(flightAngle) * flightRadius;
                    this.position.z = Math.cos(flightAngle) * flightRadius - 0.8;
                    this.position.y = 0.9 + Math.sin(simTime * 2.5) * 0.25;
                    
                    this.attitude.set(
                        -0.15,
                        flightAngle + Math.PI / 2,
                        -Math.cos(flightAngle) * 0.35
                    );
                    
                    Object.values(this.legs).forEach(l => {{
                        l.root.rotation.x = 0.5;
                    }});
                }} else if (currentBehavior === 'groom') {{
                    this.position.set(0.0, 0.16, 0.6);
                    this.attitude.set(0.05, 0.0, 0.0);
                    const rub = Math.sin(simTime * 12.0) * 0.4;
                    this.legs['LF'].root.rotation.x = 0.8 + rub;
                    this.legs['RF'].root.rotation.x = 0.8 - rub;
                }}
                
                this.group.position.copy(this.position);
                this.group.rotation.copy(this.attitude);
            }}
        }}
        
        function initFlies() {{
            flyA = new NatureFly('flyA', currentBrainMode, new THREE.Vector3(0, 0.16, -0.22), 0x10b981);
            flyB = new NatureFly('flyB', 'full', new THREE.Vector3(0.8, 0.16, -0.22), 0x0ea5e9);
            flyB.group.visible = false;
        }}
        
        // -------------------------------------------------------------
        // Camera Presets Matching User Reference Images
        // -------------------------------------------------------------
        function setCameraPreset(preset) {{
            currentCamPreset = preset;
            document.querySelectorAll('.cam-btn').forEach(b => b.classList.remove('active'));
            document.getElementById(`cam-${{preset}}`).classList.add('active');
            
            if (preset === 'feeder') {{
                camera.position.set(0.75, 0.65, 0.75);
                controls.target.set(0.0, 0.16, -0.45);
            }} else if (preset === 'portrait') {{
                camera.position.set(0.0, 0.28, 0.95);
                controls.target.set(0.0, 0.16, 0.0);
            }} else if (preset === 'plume') {{
                camera.position.set(1.8, 2.8, 2.2);
                controls.target.set(0.5, 0.1, -0.8);
            }} else if (preset === 'topdown') {{
                camera.position.set(0.0, 3.5, -0.2);
                controls.target.set(0.0, 0.0, -0.2);
            }} else if (preset === 'brain') {{
                camera.position.set(flyA.position.x + 0.2, flyA.position.y + 0.15, flyA.position.z + 0.35);
                controls.target.copy(flyA.position);
            }}
        }}
        
        // -------------------------------------------------------------
        // Controls & Behaviors
        // -------------------------------------------------------------
        function setBehavior(beh) {{
            currentBehavior = beh;
            document.querySelectorAll('.btn-grid button').forEach(b => b.classList.remove('active'));
            const btn = document.getElementById(`btn-${{beh}}`);
            if (btn) btn.classList.add('active');
            
            const badge = document.getElementById('val-state-badge');
            const mode = document.getElementById('val-loco-mode');
            if (beh === 'feed') {{
                badge.textContent = 'FEEDING ON BANANA';
                badge.style.color = '#f59e0b';
                mode.textContent = 'Proboscis Ext';
            }} else if (beh === 'roam') {{
                badge.textContent = 'WALKING TRIPOD';
                badge.style.color = '#10b981';
                mode.textContent = 'Ground Roam';
            }} else if (beh === 'fly') {{
                badge.textContent = 'FLIGHT 200 HZ';
                badge.style.color = '#0ea5e9';
                mode.textContent = 'Aerial 6-DOF';
            }} else if (beh === 'groom') {{
                badge.textContent = 'HEAD GROOMING';
                badge.style.color = '#8b5cf6';
                mode.textContent = 'Antennal Rub';
            }}
        }}
        
        function setBrainMode(m) {{
            currentBrainMode = m;
            document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
            
            if (m === 'dual') {{
                document.getElementById('mode-dual').classList.add('active');
                flyA.group.position.set(-0.6, 0.16, -0.22);
                flyB.group.position.set(0.6, 0.16, -0.22);
                flyA.switchBrain('min');
                flyB.switchBrain('full');
                flyB.group.visible = true;
                
                document.getElementById('val-neuron-count').textContent = '148 vs 2,300';
                document.getElementById('val-synapse-count').textContent = '1,095 vs 57,354';
                document.getElementById('val-power').innerHTML = '0.037 <span class="metric-unit">vs 0.575 μW</span>';
            }} else {{
                flyB.group.visible = false;
                flyA.group.position.set(0, 0.16, -0.22);
                if (m === 'min') {{
                    document.getElementById('mode-single-min').classList.add('active');
                    flyA.switchBrain('min');
                    document.getElementById('val-neuron-count').textContent = '148';
                    document.getElementById('val-synapse-count').textContent = '1,095';
                    document.getElementById('val-power').innerHTML = '0.037 <span class="metric-unit">μW</span>';
                    document.getElementById('val-retention').textContent = '100% Retained (-93.6%)';
                }} else {{
                    document.getElementById('mode-single-full').classList.add('active');
                    flyA.switchBrain('full');
                    document.getElementById('val-neuron-count').textContent = '2,300';
                    document.getElementById('val-synapse-count').textContent = '57,354';
                    document.getElementById('val-power').innerHTML = '0.575 <span class="metric-unit">μW</span>';
                    document.getElementById('val-retention').textContent = 'Original Baseline';
                }}
            }}
        }}
        
        function updateCuticleOpacity(v) {{
            cuticleOpacity = v / 100.0;
            flyA.updateCuticle(cuticleOpacity);
            if (flyB) flyB.updateCuticle(cuticleOpacity);
            document.getElementById('badge-cuticle').textContent = `${{v}}% (${{v < 25 ? 'Ghost' : v < 60 ? 'X-Ray' : 'Solid'}})`;
        }}
        
        function updateBrainGlow(v) {{
            brainGlow = v / 100.0;
            flyA.updateGlow(brainGlow);
            if (flyB) flyB.updateGlow(brainGlow);
            document.getElementById('badge-glow').textContent = `${{v}}%`;
        }}
        
        function toggleOdorPlume() {{
            odorPlumeGroup.visible = !odorPlumeGroup.visible;
            document.getElementById('btn-plume').classList.toggle('active', odorPlumeGroup.visible);
        }}
        
        function toggleStones() {{
            riverStones.forEach(s => s.visible = !s.visible);
            const vis = riverStones[0].visible;
        let lastTime = 0;
        function animate(now) {{
            requestAnimationFrame(animate);
            if (!lastTime) lastTime = now;
            const dt = Math.min(Math.max((now - lastTime) / 1000.0, 0.001), 0.1);
            lastTime = now;
            simTime += dt;
            
            flyA.step(dt);
            if (currentBrainMode === 'dual' && flyB.group.visible) {{
                flyB.step(dt);
            }}
            
            if (odorPlumeGroup.visible) {{
                const pos = odorPlumeGroup.children[0].geometry.attributes.position.array;
                for (let i = 0; i < pos.length; i += 3) {{
                    pos[i+2] += dt * 0.4;
                    if (pos[i+2] > 3.0) {{
                        pos[i+2] = -0.65;
                        pos[i] = (Math.random() - 0.5) * 0.3;
                    }}
                }}
                odorPlumeGroup.children[0].geometry.attributes.position.needsUpdate = true;
            }}
            
            controls.update();
            renderer.render(scene, camera);
        }}
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    build_nature_arena()
