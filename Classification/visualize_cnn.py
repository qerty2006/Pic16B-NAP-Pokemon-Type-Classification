#!/usr/bin/env python3
"""
Hooks into the trained EfficientNet-B0 model to visualize intermediate
convolutional feature maps, learned color filters, and a decomposed
view of per-color-channel convolutions.
"""
import sys
from pathlib import Path
import random

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Ensure local classification directory is in python path
classification_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(classification_dir))

from dataset import PokemonSpriteDataset, DEFAULT_TRANSFORM, TYPES, gen_stratified_split, rgba_to_rgb
from cnn_model import build_model

# Dictionary to hold the captured feature maps
activations = {}

def get_activation(name):
    """PyTorch hook to capture the output of a specific layer."""
    def hook(model, input, output):
        # Output is a tensor of shape (batch, channels, height, width)
        activations[name] = output.detach()
    return hook

def main():
    checkpoint_path = classification_dir / "checkpoints" / "best.pt"
    if not checkpoint_path.exists():
        print(f"Error: No checkpoint found at {checkpoint_path}. Train the model first.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    model = build_model(num_classes=len(TYPES)).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # 2. Register Hooks on EfficientNet-B0
    model.features[4].register_forward_hook(get_activation('Middle (Textures/Patterns)'))
    model.features[8].register_forward_hook(get_activation('Final (High-Level Concepts)'))

    # 3. Get a random test image
    dataset = PokemonSpriteDataset(transform=DEFAULT_TRANSFORM)
    _, _, test_idx = gen_stratified_split(dataset.index)
    
    sample_idx = random.choice(test_idx)
    img_path, raw_label = dataset.index[sample_idx]
    
    # Load raw image for displaying alongside the maps
    raw_img = rgba_to_rgb(Image.open(img_path).convert("RGBA"))
    
    # Preprocess for the model
    tensor_img = DEFAULT_TRANSFORM(raw_img).unsqueeze(0).to(device)

    # 4. Forward Pass
    with torch.no_grad():
        logits = model(tensor_img)
        probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        # === DECOMPOSED PER-CHANNEL CONVOLUTION ===
        # Get the very first convolution layer and its weights
        conv1 = model.features[0][0]
        W = conv1.weight.detach() # Shape: [32, 3, 3, 3]
        stride = conv1.stride
        padding = conv1.padding
        
        # Split input image by channel [1, 1, H, W]
        X_r, X_g, X_b = tensor_img[:, 0:1], tensor_img[:, 1:2], tensor_img[:, 2:3]
        
        # Split weights by channel [32, 1, 3, 3]
        W_r, W_g, W_b = W[:, 0:1], W[:, 1:2], W[:, 2:3]
        
        # Run isolated 2D convolutions manually
        out_r = F.conv2d(X_r, W_r, stride=stride, padding=padding)[0].cpu().numpy()
        out_g = F.conv2d(X_g, W_g, stride=stride, padding=padding)[0].cpu().numpy()
        out_b = F.conv2d(X_b, W_b, stride=stride, padding=padding)[0].cpu().numpy()
        out_sum = out_r + out_g + out_b  # Direct linear sum of the color channels
    
    # Decode predictions
    true_types = [TYPES[j] for j, val in enumerate(raw_label) if val == 1.0]
    
    print(f"Evaluating: {img_path.name}")
    print(f"True Types: {true_types}")
    print(f"Top Preds:  {[(TYPES[i], f'{probs[i]:.2%}') for i in np.argsort(probs)[::-1][:3]]}")

    # 5. Plot the Results
    num_layers = len(activations)
    maps_to_show = 8 
    
    # We now have 6 extra rows: Input, RGB Weights, 3 manual channel convolutions, and their sum
    fig, axes = plt.subplots(num_layers + 6, maps_to_show, figsize=(16, 2.5 * (num_layers + 6)))
    fig.suptitle(f"Neural Firings for: {img_path.name}\nTrue: {true_types}", fontsize=16)

    # --- ROW 0: Original Image & R, G, B Split ---
    ax_orig = axes[0, 0]
    ax_orig.imshow(raw_img)
    ax_orig.set_title("Original Image")
    ax_orig.axis("off")

    r_img, g_img, b_img = raw_img.split()
    channels = [(r_img, 'Red', 'Reds'), (g_img, 'Green', 'Greens'), (b_img, 'Blue', 'Blues')]
    
    for i, (ch_img, name, cmap) in enumerate(channels):
        ax = axes[0, i + 1]
        ax.imshow(ch_img, cmap=cmap)
        ax.set_title(f"Input: {name} Channel")
        ax.axis("off")

    for i in range(4, maps_to_show):
        axes[0, i].axis("off")

    # --- ROW 1: Learned Color Filters (First Conv Layer Weights) ---
    first_conv_weights = W.cpu().numpy()
    
    for map_idx in range(maps_to_show):
        ax = axes[1, map_idx]
        if map_idx < first_conv_weights.shape[0]:
            w = first_conv_weights[map_idx] 
            
            # Normalize to [0, 1] for visual RGB plotting
            w_min, w_max = w.min(), w.max()
            w_norm = (w - w_min) / (w_max - w_min + 1e-8)
            w_norm = np.transpose(w_norm, (1, 2, 0))
            
            ax.imshow(w_norm, interpolation='nearest')
            ax.set_title(f"Filter {map_idx} (RGB Weights)", fontsize=10)
        ax.axis('off')

    # --- ROWS 2, 3, 4, 5: Isolated Channel Convolutions & Their Sum ---
    # To mathematically visualize the sum correctly, we must lock the colormap scale (vmin, vmax)
    # across the R, G, B, and Sum rows. Otherwise, Matplotlib auto-scales each subplot 
    # independently based on its own min/max, which breaks the visual linear addition.
    # We use a diverging colormap ('coolwarm') centered at 0 so positive (red) and negative (blue)
    # firings sum together intuitively, with 0 being white.
    
    for map_idx in range(maps_to_show):
        if map_idx < out_sum.shape[0]:
            # Find absolute max magnitude to center the colormap at 0 for this specific channel
            c_max = max(
                np.abs(out_r[map_idx]).max(), 
                np.abs(out_g[map_idx]).max(), 
                np.abs(out_b[map_idx]).max(), 
                np.abs(out_sum[map_idx]).max()
            )
            
            # Plot Red Conv
            ax = axes[2, map_idx]
            ax.imshow(out_r[map_idx], cmap='coolwarm', vmin=-c_max, vmax=c_max)
            if map_idx == 0: ax.set_title(f"Red Conv ($X_R \\star W_R$)\nCh {map_idx}", fontsize=10)
            else: ax.set_title(f"Ch {map_idx}", fontsize=10)
            ax.axis('off')

            # Plot Green Conv
            ax = axes[3, map_idx]
            ax.imshow(out_g[map_idx], cmap='coolwarm', vmin=-c_max, vmax=c_max)
            if map_idx == 0: ax.set_title(f"Green Conv ($X_G \\star W_G$)\nCh {map_idx}", fontsize=10)
            else: ax.set_title(f"Ch {map_idx}", fontsize=10)
            ax.axis('off')

            # Plot Blue Conv
            ax = axes[4, map_idx]
            ax.imshow(out_b[map_idx], cmap='coolwarm', vmin=-c_max, vmax=c_max)
            if map_idx == 0: ax.set_title(f"Blue Conv ($X_B \\star W_B$)\nCh {map_idx}", fontsize=10)
            else: ax.set_title(f"Ch {map_idx}", fontsize=10)
            ax.axis('off')

            # Plot Sum
            ax = axes[5, map_idx]
            ax.imshow(out_sum[map_idx], cmap='coolwarm', vmin=-c_max, vmax=c_max)
            if map_idx == 0: ax.set_title(f"Sum of RGB (Pre-Activation)\nCh {map_idx}", fontsize=10)
            else: ax.set_title(f"Ch {map_idx}", fontsize=10)
            ax.axis('off')
        else:
            for row in range(2, 6):
                axes[row, map_idx].axis('off')

    # --- ROWS 6+: Feature Maps (Neuron Activations) ---
    for row_idx, (layer_name, activation_tensor) in enumerate(activations.items(), start=6):
        act = activation_tensor[0].cpu().numpy()
        
        for map_idx in range(maps_to_show):
            ax = axes[row_idx, map_idx]
            if map_idx < act.shape[0]:
                ax.imshow(act[map_idx], cmap='viridis')
                ax.axis('off')
                if map_idx == 0:
                    ax.set_title(f"{layer_name}\nCh {map_idx}", fontsize=10)
                else:
                    ax.set_title(f"Ch {map_idx}", fontsize=10)
            else:
                ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    
    save_path = classification_dir / "results" / f"activation_decomposed_{img_path.stem}.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    print(f"\nSaved decomposed visualization to {save_path}")
    plt.show()

if __name__ == "__main__":
    main()
