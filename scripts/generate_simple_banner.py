"""
Generate a clean, minimalist, publication-grade black & white / dark mode banner
for OrthoCop GitHub README.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image, ImageEnhance, ImageOps
import numpy as np

def create_banner():
    # Dimensions: 1400 x 680 pixels, dpi=100 -> 14 x 6.8 inches
    fig, ax = plt.subplots(figsize=(14, 6.8), dpi=100)
    bg_color = '#0b0e14' # Dark charcoal / slate
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.set_xlim(0, 1400)
    ax.set_ylim(0, 680)
    ax.axis('off')

    # Load Fly image and convert to clean black-and-white
    fly_img = Image.open('visualizations/mujoco_walker_hero.png').convert('L')
    fly_arr = np.array(fly_img)
    
    # Enhance contrast to make the fly pop cleanly against dark background
    fly_arr = np.clip((fly_arr.astype(float) - 10) * 1.35, 0, 255).astype(np.uint8)
    
    # Crop fly tightly
    # Find bounding box where intensity > 15
    mask = fly_arr > 15
    coords = np.argwhere(mask)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0)
    fly_cropped = fly_arr[y0:y1, x0:x1]
    
    # Place Fly on the right half: x in [740, 1360], y in [80, 600]
    # We display it using imshow
    ax.imshow(fly_cropped, cmap='gray', extent=[720, 1360, 60, 600], aspect='auto', zorder=2)
    
    # Add subtle technical callout lines and labels for fly anatomy
    # Wing tip
    ax.annotate("Flapping Wings\n(Blade-Element Aerodynamics)", 
                xy=(820, 500), xytext=(730, 610),
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.2),
                color="#c9d1d9", fontsize=9, fontweight='medium', family='sans-serif',
                ha='right', va='bottom', zorder=3)
    
    # Thorax / Muscles
    ax.annotate("Flight Power Circuits\n(DLM / DVM Resonant Muscles)", 
                xy=(1020, 340), xytext=(1260, 430),
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.2),
                color="#c9d1d9", fontsize=9, fontweight='medium', family='sans-serif',
                ha='left', va='bottom', zorder=3)
                
    # Haltere
    ax.annotate("Haltere Mechanoreceptors\n(Coriolis Gyroscope @ 200 Hz)", 
                xy=(1070, 260), xytext=(1250, 170),
                arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.2),
                color="#c9d1d9", fontsize=9, fontweight='medium', family='sans-serif',
                ha='left', va='top', zorder=3)

    # Left Side: Typography & Clean Architecture Diagram
    # Title
    ax.text(60, 600, "ORTHOCOP", color="#ffffff", fontsize=30, fontweight='heavy', family='sans-serif', zorder=3)
    ax.text(60, 565, "Embodied Drosophila Connectome Flight Controller", color="#8b949e", fontsize=13, fontweight='medium', family='sans-serif', zorder=3)
    
    # Subtle horizontal line under title
    ax.plot([60, 680], [545, 545], color="#30363d", lw=1.2, zorder=3)
    
    # Diagram Blocks
    boxes = [
        ("1. SENSORY FEEDBACK", "Haltere Coriolis strain, antennal airflow, and ground contact mechanoreceptors", 460),
        ("2. SPIKING CONNECTOME", "10,668 Leaky Integrate-and-Fire neurons (Brian2) | 527K synaptic connections", 365),
        ("3. BIOPHYSICAL MOTOR DECODER", "Neural firing rates (Hz) mapped to 12-DOF wing stroke, roll, & pitch torques", 270),
        ("4. MULTI-BODY AERODYNAMICS", "Google DeepMind FlyBody in MuJoCo @ 5,000 Hz | Closed-loop 1.0x real-time", 175)
    ]
    
    box_w = 620
    box_h = 65
    
    for i, (title, desc, y_pos) in enumerate(boxes):
        # Draw rounded rectangle
        rect = patches.FancyBboxPatch(
            (60, y_pos), box_w, box_h,
            boxstyle="round,pad=0.0,rounding_size=6",
            linewidth=1.2, edgecolor="#30363d", facecolor="#161b22", zorder=3
        )
        ax.add_patch(rect)
        
        # Text inside box
        ax.text(80, y_pos + 40, title, color="#58a6ff" if i==1 else ("#3fb950" if i==2 else "#f0f6fc"), fontsize=10.5, fontweight='bold', family='sans-serif', zorder=4)
        ax.text(80, y_pos + 18, desc, color="#8b949e", fontsize=8.5, family='sans-serif', zorder=4)
        
        # Connecting downward arrow between boxes
        if i < len(boxes) - 1:
            ax.annotate("", xy=(370, y_pos - 18), xytext=(370, y_pos),
                        arrowprops=dict(arrowstyle="->", color="#58a6ff", lw=1.5),
                        zorder=4)

    # Bottom Pill Badges
    pills = [
        "10,668 Neurons",
        "527K Synapses",
        "5,000 Hz MuJoCo",
        "FPGA Edge Target"
    ]
    
    px = 60
    for pill in pills:
        pw = len(pill) * 9.5 + 24
        prect = patches.FancyBboxPatch(
            (px, 70), pw, 32,
            boxstyle="round,pad=0.0,rounding_size=16",
            linewidth=1.0, edgecolor="#30363d", facecolor="#1c2128", zorder=3
        )
        ax.add_patch(prect)
        ax.text(px + pw/2, 86, pill, color="#c9d1d9", fontsize=9, fontweight='semibold', family='sans-serif', ha='center', va='center', zorder=4)
        px += pw + 15

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    output_path = 'assets/orthocop_banner.png'
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor='none', dpi=100)
    plt.close()
    print(f"[+] Clean black & white technical banner successfully generated: {output_path}")

if __name__ == '__main__':
    create_banner()
