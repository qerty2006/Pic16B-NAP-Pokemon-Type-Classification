#!/usr/bin/env python3
"""
Hooks into the trained EfficientNet-B0 model to visualize intermediate
convolutional feature maps, learned color filters, and a decomposed
view of per-color-channel convolutions, outputting to a beautiful HTML dashboard.
"""
import sys
import io
import base64
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib

# Ensure local classification directory is in python path
classification_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(classification_dir))

from dataset import PokemonSpriteDataset, DEFAULT_TRANSFORM, GRAYSCALE_DEFAULT_TRANSFORM, TYPES, gen_stratified_split, rgba_to_rgb
from cnn_model import build_model

activations = {}

def get_activation(name):
    def hook(model, input, output):
        activations[name] = output.detach()
    return hook

def array_to_b64(arr, cmap_name=None, vmin=None, vmax=None):
    """Converts a numpy array to a base64 PNG string, applying an optional colormap."""
    if vmin is not None and vmax is not None:
        # Scale to [0, 1] based on explicit limits (crucial for symmetric coolwarm)
        arr_norm = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
    else:
        # Auto-scale to [0, 1]
        arr_norm = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        
    if cmap_name:
        colormap = matplotlib.colormaps.get_cmap(cmap_name)
        # colormap returns RGBA floats in [0, 1]
        rgba_img = colormap(arr_norm)
        img = Image.fromarray((rgba_img[:, :, :3] * 255).astype(np.uint8))
    else:
        # Assumes arr is already RGB [H, W, 3] in [0, 1]
        img = Image.fromarray((arr_norm * 255).astype(np.uint8))
        
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def img_to_b64(img):
    """Directly converts a PIL Image to base64 PNG."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

def overlay_activation_on_image(act_map, orig_img):
    """Upsamples an activation map and uses it to scale the original image's RGB pixels."""
    # Normalize activation map to [0, 1]
    act_min, act_max = act_map.min(), act_map.max()
    if act_max - act_min > 1e-8:
        act_norm = (act_map - act_min) / (act_max - act_min)
    else:
        act_norm = np.zeros_like(act_map)
        
    # Convert to PIL Image to easily resize (interpolate) to original image dimensions
    # Using BILINEAR interpolation to smoothly stretch the low-res matrix
    act_img = Image.fromarray((act_norm * 255).astype(np.uint8))
    
    # Use Image.Resampling.BILINEAR for modern PIL, or Image.BILINEAR for older versions
    resample_mode = getattr(Image, 'Resampling', Image).BILINEAR
    act_resized = act_img.resize(orig_img.size, resample_mode)
    
    # Convert both to float arrays in [0, 1]
    orig_arr = np.array(orig_img).astype(np.float32) / 255.0
    mask_arr = np.array(act_resized).astype(np.float32) / 255.0
    
    # Expand mask from (H, W) to (H, W, 1) for the Hadamard product (element-wise multiplication)
    mask_arr = np.expand_dims(mask_arr, axis=-1)
    
    # Multiply (scale RGB by activation signal)
    scaled_arr = orig_arr * mask_arr
    
    # Convert back to image
    return Image.fromarray((scaled_arr * 255).astype(np.uint8))


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate activation dashboards for N samples.")
    parser.add_argument("-n", "--n-samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert the dataset to grayscale and load the grayscale model"
    )
    args = parser.parse_args()

    checkpoint_dir = classification_dir / "checkpoints"
    checkpoint_path = None
    if args.grayscale:
        checkpoint_path = checkpoint_dir / "best_grayscale.pt"
        if not checkpoint_path.exists():
            checkpoint_path = checkpoint_dir / "best.pt"
    else:
        for name in ["best_color.pt", "best.pt", "best_grayscale.pt"]:
            p = checkpoint_dir / name
            if p.exists():
                checkpoint_path = p
                break

    if not checkpoint_path or not checkpoint_path.exists():
        print(f"Error: No checkpoint found at {checkpoint_path}. Train the model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model & Register Hooks
    model = build_model(num_classes=len(TYPES)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    is_grayscale = args.grayscale or ckpt.get("args", {}).get("grayscale", False)
    suffix = "grayscale" if is_grayscale else "color"
    print(f"Loaded checkpoint {checkpoint_path.name} (grayscale mode: {is_grayscale})")

    model.features[4].register_forward_hook(get_activation('Middle (Textures/Patterns)'))
    model.features[8].register_forward_hook(get_activation('Final (High-Level Concepts)'))

    # 2. Get random image & run forward pass
    active_transform = GRAYSCALE_DEFAULT_TRANSFORM if is_grayscale else DEFAULT_TRANSFORM
    dataset = PokemonSpriteDataset(transform=active_transform)
    _, _, test_idx = gen_stratified_split(dataset.index)
    
    n_samples = min(args.n_samples, len(test_idx))
    chosen_samples = random.sample(test_idx, n_samples)
    
    save_dir = classification_dir / "results" / f"activation_dashboards_{suffix}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Generating {n_samples} dashboards in {save_dir}...\n")

    html_links = []

    for count, sample_idx in enumerate(chosen_samples, 1):
        img_path, raw_label = dataset.index[sample_idx]
        pokemon_folder = img_path.parent.name
        print(f"[{count}/{n_samples}] Processing Pokemon {pokemon_folder} ({img_path.name})...")
        
        raw_img = rgba_to_rgb(Image.open(img_path).convert("RGBA"))
        if is_grayscale:
            raw_img = raw_img.convert("L").convert("RGB")
        tensor_img = active_transform(raw_img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor_img)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
            
            # Decomposed Convolutions
            conv1 = model.features[0][0]
            W = conv1.weight.detach()
            stride, padding = conv1.stride, conv1.padding
            
            X_r, X_g, X_b = tensor_img[:, 0:1], tensor_img[:, 1:2], tensor_img[:, 2:3]
            W_r, W_g, W_b = W[:, 0:1], W[:, 1:2], W[:, 2:3]
            
            out_r = F.conv2d(X_r, W_r, stride=stride, padding=padding)[0].cpu().numpy()
            out_g = F.conv2d(X_g, W_g, stride=stride, padding=padding)[0].cpu().numpy()
            out_b = F.conv2d(X_b, W_b, stride=stride, padding=padding)[0].cpu().numpy()
        
        true_types = [TYPES[j] for j, val in enumerate(raw_label) if val == 1.0]
        pred_types = [(TYPES[i], f'{probs[i]:.1%}') for i in np.argsort(probs)[::-1][:3]]

        maps_to_show = 8
        
        # 3. Prepare Image Data for HTML
        img_b64_orig = img_to_b64(raw_img)
        if not is_grayscale:
            r, g, b = raw_img.split()
            z = Image.new("L", r.size, 0)
            img_b64_r = img_to_b64(Image.merge("RGB", (r, z, z)))
            img_b64_g = img_to_b64(Image.merge("RGB", (z, g, z)))
            img_b64_b = img_to_b64(Image.merge("RGB", (z, z, b)))

        # HTML Generation
        html = [f"""<!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Neural Firings: {pokemon_folder}</title>
            <style>
                :root {{
                    --bg: #0f172a;
                    --panel: #1e293b;
                    --text: #f8fafc;
                    --accent: #38bdf8;
                    --border: #334155;
                }}
                body {{
                    background-color: var(--bg);
                    color: var(--text);
                    font-family: 'Segoe UI', system-ui, sans-serif;
                    margin: 0;
                    padding: 30px;
                    overflow-x: hidden;
                }}
                .header {{
                    text-align: center;
                    padding-bottom: 20px;
                    border-bottom: 2px solid var(--border);
                    margin-bottom: 30px;
                }}
                h1 {{ margin: 0 0 10px 0; color: var(--accent); font-weight: 300; letter-spacing: 1px; }}
                .metadata {{ font-size: 1.1em; color: #94a3b8; }}
                .badge {{
                    background: rgba(56, 189, 248, 0.15);
                    color: var(--accent);
                    padding: 4px 10px;
                    border-radius: 6px;
                    font-weight: 600;
                    font-size: 0.9em;
                    margin: 0 5px;
                }}
                
                .section-title {{
                    font-size: 1.4em;
                    margin: 40px 0 15px 0;
                    padding-left: 10px;
                    border-left: 4px solid var(--accent);
                }}
                
                .grid-container {{
                    display: grid;
                    grid-template-columns: 200px repeat({maps_to_show}, 1fr);
                    gap: 15px;
                    align-items: center;
                    margin-bottom: 15px;
                    background: var(--panel);
                    padding: 15px;
                    border-radius: 12px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                }}
                
                .row-label {{
                    font-weight: 600;
                    font-size: 0.95em;
                    color: #cbd5e1;
                    line-height: 1.4;
                }}
                .math-text {{
                    font-family: 'Courier New', Courier, monospace;
                    color: #f472b6; /* pink for math */
                    font-size: 0.9em;
                    display: block;
                    margin-top: 4px;
                }}
                
                .card {{
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                }}
                .card-title {{
                    font-size: 0.75em;
                    color: #94a3b8;
                    margin-bottom: 8px;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }}
                .img-wrapper {{
                    width: 100%;
                    aspect-ratio: 1 / 1;
                    background: #000;
                    border-radius: 8px;
                    overflow: hidden;
                    border: 1px solid var(--border);
                }}
                .img-wrapper img {{
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                    image-rendering: pixelated;
                    transition: transform 0.3s ease;
                }}
                .img-wrapper img:hover {{
                    transform: scale(1.1);
                }}
                
                .input-grid {{
                    display: grid;
                    grid-template-columns: repeat({"1" if is_grayscale else "4"}, 200px);
                    gap: 20px;
                    justify-content: center;
                    margin-bottom: 40px;
                }}
                .input-card img {{ width: 100%; border-radius: 12px; border: 2px solid var(--border); }}
                .nav-links {{ margin-top: 15px; font-size: 0.9em; }}
                .nav-links a {{ color: var(--accent); text-decoration: none; margin: 0 10px; }}
                .nav-links a:hover {{ text-decoration: underline; }}
            </style>
        </head>
        <body>

        <div class="header">
            <h1>Activation Dashboard</h1>
            <div class="metadata">
                Pokemon ID/Form: <span style="color:#fff">{pokemon_folder}</span> &nbsp;|&nbsp; 
                True Type: { "".join([f'<span class="badge">{t}</span>' for t in true_types]) }
            </div>
            <div style="margin-top:10px; font-size:0.9em; color:#64748b;">
                Top Preds: {', '.join([f'{t[0]} ({t[1]})' for t in pred_types])}
            </div>
            <div class="nav-links">
                <a href="index.html">&larr; Back to Index</a>
            </div>
        </div>
        """]

        # Inputs Section
        if is_grayscale:
            html.append(f"""
            <div class="section-title">Network Input (Grayscale)</div>
            <div class="input-grid">
                <div class="card"><div class="card-title">Grayscale Input</div><img src="data:image/png;base64,{img_b64_orig}"></div>
            </div>
            """)
        else:
            html.append(f"""
            <div class="section-title">Network Input (Split Channels)</div>
            <div class="input-grid">
                <div class="card"><div class="card-title">Original RGB</div><img src="data:image/png;base64,{img_b64_orig}"></div>
                <div class="card"><div class="card-title">Red Channel Input</div><img src="data:image/png;base64,{img_b64_r}"></div>
                <div class="card"><div class="card-title">Green Channel Input</div><img src="data:image/png;base64,{img_b64_g}"></div>
                <div class="card"><div class="card-title">Blue Channel Input</div><img src="data:image/png;base64,{img_b64_b}"></div>
            </div>
            """)

        # Explanation
        if is_grayscale:
            html.append("""
            <div class="section-title">Layer 1: Convolutions</div>
            <p style="color:#94a3b8; font-size:0.95em; max-width:800px; margin-bottom: 20px;">
                Visualizing the first layer filters and the resulting output signal. The filters (weights) are 3x3x3 matrices.
                Since the input is grayscale, the color channels are identical. We show the combined activation signal response below.
            </p>
            """)
        else:
            html.append("""
            <div class="section-title">Layer 1: Color Basis Convolutions</div>
            <p style="color:#94a3b8; font-size:0.95em; max-width:800px; margin-bottom: 20px;">
                Visualizing the exact linear algebra of the first layer. The filters (weights) are 3x3x3 matrices. 
                We separate the 2D cross-correlation for each color channel, then composite them into an RGB image. 
                <span style="color:#f472b6;">For the individual channel rows: Red = Positive firing, Blue = Negative firing.</span>
            </p>
            """)
        
        W_np = W.cpu().numpy()
        
        # 1. Filters Row
        html.append('<div class="grid-container"><div class="row-label">Learned Filters<span class="math-text">W ∈ R³ˣ³</span></div>')
        for map_idx in range(maps_to_show):
            w = W_np[map_idx]
            w_norm = np.transpose(w, (1, 2, 0)) # H, W, C
            b64 = array_to_b64(w_norm)
            html.append(f'<div class="card"><div class="card-title">Filter {map_idx}</div><div class="img-wrapper"><img src="data:image/png;base64,{b64}"></div></div>')
        html.append('</div>')

        # 2. Decomposed Convolutions Rows (Coolwarm) - Omitted in grayscale
        if not is_grayscale:
            rows = [
                (out_r, 'Red Convolution', 'X_R ★ W_R'),
                (out_g, 'Green Convolution', 'X_G ★ W_G'),
                (out_b, 'Blue Convolution', 'X_B ★ W_B')
            ]
            
            for out_data, label, math_lbl in rows:
                html.append(f'<div class="grid-container"><div class="row-label">{label}<span class="math-text">{math_lbl}</span></div>')
                for map_idx in range(maps_to_show):
                    # Find max magnitude across all 3 components for THIS filter to keep the color scale mathematically sound
                    c_max = max(
                        np.abs(out_r[map_idx]).max(), 
                        np.abs(out_g[map_idx]).max(), 
                        np.abs(out_b[map_idx]).max()
                    )
                    b64 = array_to_b64(out_data[map_idx], cmap_name='coolwarm', vmin=-c_max, vmax=c_max)
                    html.append(f'<div class="card"><div class="card-title">Ch {map_idx}</div><div class="img-wrapper"><img src="data:image/png;base64,{b64}"></div></div>')
                html.append('</div>')

        # 2.5. Composite Row
        if is_grayscale:
            html.append('<div class="grid-container"><div class="row-label">Signal Composite<span class="math-text">Composite = max(0, ∑ X_C ★ W_C)</span></div>')
        else:
            html.append('<div class="grid-container"><div class="row-label">RGB Signal Composite<span class="math-text">RGB = max(0, X_C ★ W_C)</span></div>')
            
        for map_idx in range(maps_to_show):
            r_sig = out_r[map_idx]
            g_sig = out_g[map_idx]
            b_sig = out_b[map_idx]
            
            if is_grayscale:
                # Sum the signal from the 3 identical grayscale input channels
                sig = r_sig + g_sig + b_sig
                sig = np.clip(sig, 0, None)
                s_max = sig.max()
                if s_max > 0:
                    sig_norm = sig / s_max
                else:
                    sig_norm = sig
                img = Image.fromarray((sig_norm * 255).astype(np.uint8))
            else:
                # Stack into [H, W, 3] and apply ReLU (clip negatives to 0)
                # We only want to plot positive signal contributions as color brightness.
                rgb_stack = np.stack([r_sig, g_sig, b_sig], axis=-1)
                rgb_stack = np.clip(rgb_stack, 0, None)
                
                # Normalize to [0, 1] based on the max signal across all 3 channels
                s_max = rgb_stack.max()
                if s_max > 0:
                    rgb_norm = rgb_stack / s_max
                else:
                    rgb_norm = rgb_stack
                img = Image.fromarray((rgb_norm * 255).astype(np.uint8))
                
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            
            html.append(f'<div class="card"><div class="card-title">Ch {map_idx}</div><div class="img-wrapper"><img src="data:image/png;base64,{b64}"></div></div>')
        html.append('</div>')


        # 3. Deep Features Rows
        html.append('<div class="section-title" style="margin-top:50px;">Deeper Network Activations (Spatial Attention Overlay)</div>')
        html.append('<p style="color:#94a3b8; font-size:0.95em; max-width:800px; margin-bottom: 20px;">')
        html.append('Here, the lower-resolution activation matrices are upsampled and used as a multiplier ')
        html.append('on the original image. <span style="color:#f472b6;">Bright pixels = high neural signal; Black pixels = ignored regions.</span></p>')
        
        for layer_name, act_tensor in activations.items():
            act = act_tensor[0].cpu().numpy()
            html.append(f'<div class="grid-container"><div class="row-label">{layer_name}</div>')
            for map_idx in range(maps_to_show):
                # Apply our new interpolation and multiplication logic instead of colormaps!
                overlay_img = overlay_activation_on_image(act[map_idx], raw_img)
                b64 = img_to_b64(overlay_img)
                html.append(f'<div class="card"><div class="card-title">Ch {map_idx}</div><div class="img-wrapper"><img src="data:image/png;base64,{b64}"></div></div>')
            html.append('</div>')

        html.append("</body></html>")

        # Write to file
        filename = f"dashboard_{pokemon_folder}.html"
        save_path = save_dir / filename
        save_path.write_text("".join(html), encoding="utf-8")
        
        # Store link for index
        true_types_str = " / ".join(true_types)
        html_links.append((filename, pokemon_folder, true_types_str))

    # Create an index.html file to link all generated dashboards
    print("\nGenerating index.html...")
    index_html = [f"""<!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Activation Dashboards Index</title>
        <style>
            body {{ background: #0f172a; color: #f8fafc; font-family: 'Segoe UI', sans-serif; padding: 40px; }}
            h1 {{ color: #38bdf8; border-bottom: 2px solid #334155; padding-bottom: 10px; }}
            ul {{ list-style-type: none; padding: 0; }}
            li {{ background: #1e293b; margin: 10px 0; padding: 15px; border-radius: 8px; border: 1px solid #334155; }}
            a {{ color: #38bdf8; text-decoration: none; font-weight: bold; font-size: 1.1em; }}
            a:hover {{ text-decoration: underline; }}
            .type-badge {{ float: right; background: rgba(56, 189, 248, 0.15); color: #38bdf8; padding: 4px 10px; border-radius: 6px; font-size: 0.85em; }}
        </style>
    </head>
    <body>
        <h1>Generated Activation Dashboards</h1>
        <p>Select a sample below to view its neural firings, linear algebra distributions, and attention maps.</p>
        <ul>
    """]
    
    for file_name, img_name, types in html_links:
        index_html.append(f'<li><span class="type-badge">{types}</span><a href="{file_name}">Dashboard for {img_name}</a></li>')
        
    index_html.append("</ul></body></html>")
    
    index_path = save_dir / "index.html"
    index_path.write_text("".join(index_html), encoding="utf-8")
    
    print(f"Done! View your results at: {index_path}")

if __name__ == "__main__":
    main()