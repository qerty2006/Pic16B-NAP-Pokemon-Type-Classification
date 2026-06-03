#!/usr/bin/env python3
"""
General Pipeline for training, evaluating, and viewing results on the Pokemon Type Classification task.
This script coordinates:
1. Preparing splits of the dataset (supporting stratified split and generation-based split)
2. Training the CNN model (EfficientNet-V2-S by default) with support for test runs (2 epochs)
3. Evaluating the trained model on the test split
4. Visualizing results (producing detailed interactive HTML galleries of the classification results)

Supports fully hotswapping dataset splits, paths, checkpoints, and output results directories.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

# Ensure local classification directory is in python path
classification_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(classification_dir))
sys.path.insert(0, str(classification_dir / "Patrick"))

from dataset import (
    PokemonSpriteDataset,
    TYPES,
    gen_stratified_split,
    gen_gen_split,
    TRAIN_TRANSFORM,
    DEFAULT_TRANSFORM,
    GRAYSCALE_TRAIN_TRANSFORM,
    GRAYSCALE_DEFAULT_TRANSFORM,
    parse_folder_id,
    get_generation
)
from cnn_model import build_model
from train import make_weighted_sampler, run_epoch2, log
from evaluate import print_summary, print_per_gen_metrics, img_to_b64
from Visualizer import save_all_mistake_examples_with_probs

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Pipeline for Pokemon Type Classification (Train, Eval, and Visualize)"
    )
    
    # Mode configurations
    parser.add_argument(
        "--mode", 
        type=str, 
        default="all", 
        choices=["all", "prepare", "train", "evaluate", "visualize"],
        help="Pipeline step to run: 'prepare', 'train', 'evaluate', 'visualize', or 'all' (default)"
    )
    
    # Dataset and split settings (for hotswapping)
    parser.add_argument(
        "--split", 
        type=str, 
        default="stratified", 
        choices=["stratified", "generation"],
        help="Type of split strategy to use: 'stratified' (default) or 'generation'"
    )
    parser.add_argument(
        "--val-frac", 
        type=float, 
        default=0.15, 
        help="Fraction of validation samples (default: 0.15)"
    )
    parser.add_argument(
        "--test-frac", 
        type=float, 
        default=0.15, 
        help="Fraction of test samples for stratified split (default: 0.15)"
    )
    parser.add_argument(
        "--train-gens", 
        type=int, 
        nargs="+", 
        default=[1, 2, 3], 
        help="Generations to train on for 'generation' split (default: 1 2 3)"
    )
    parser.add_argument(
        "--test-gens", 
        type=int, 
        nargs="+", 
        default=[4, 5, 6], 
        help="Generations to test on for 'generation' split (default: 4 5 6)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for splitting data (default: 42)"
    )
    
    # Training configurations
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Number of epochs to train (default: 30)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4, 
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--freeze-backbone", 
        action="store_true", 
        help="Freeze backbone and only train the classifier head"
    )
    parser.add_argument(
        "--test-run", 
        action="store_true", 
        help="Run a quick test training of only 2 epochs"
    )
    parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert the dataset to grayscale before passing it to the model"
    )
    
    # File paths and Hotswappability options
    parser.add_argument(
        "--results-dir", 
        type=str, 
        default=None, 
        help="Directory to save evaluation results and galleries (defaults to classification/results)"
    )
    parser.add_argument(
        "--checkpoint-dir", 
        type=str, 
        default=None, 
        help="Directory to save checkpoint best.pt (defaults to classification/checkpoints)"
    )
    parser.add_argument(
        "--checkpoint-name", 
        type=str, 
        default="best.pt", 
        help="Name of the checkpoint file (default: best.pt)"
    )
    
    return parser.parse_args()

def setup_directories(args):
    # Set default values if not specified
    results_dir = Path(args.results_dir) if args.results_dir else classification_dir / "results"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else classification_dir / "checkpoints"
    
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    return results_dir, checkpoint_dir

def get_splits(args, dataset):
    if args.split == "stratified":
        print(f"Generating stratified split with val_frac={args.val_frac}, test_frac={args.test_frac}, seed={args.seed}...")
        train_idx, val_idx, test_idx = gen_stratified_split(
            dataset.index, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.seed
        )
    elif args.split == "generation":
        print(f"Generating generation-based split (train gens {args.train_gens}, test gens {args.test_gens}, val_frac={args.val_frac}, seed={args.seed})...")
        train_idx, val_idx, test_idx = gen_gen_split(
            dataset.index, train_gens=args.train_gens, val_frac=args.val_frac, test_gens=args.test_gens, seed=args.seed
        )
    else:
        raise ValueError(f"Unknown split type: {args.split}")
    
    print(f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx

def collect_predictions_pipeline(model, loader, device):
    """Collect predictions from evaluation loader using the GAP_THRESHOLD logic in evaluate.py."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    GAP_THRESHOLD = 0.25

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            preds = np.zeros_like(probs, dtype=int)

            for i in range(len(probs)):
                sorted_idx = np.argsort(probs[i])[::-1]
                # Guarantee absolute #1 highest guess
                preds[i, sorted_idx[0]] = 1
                # Check if the 2nd highest guess is within the gap threshold
                if (probs[i, sorted_idx[0]] - probs[i, sorted_idx[1]]) < GAP_THRESHOLD:
                    preds[i, sorted_idx[1]] = 1

            all_labels.append(labels.int().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    return (
        np.vstack(all_labels),
        np.vstack(all_preds),
        np.vstack(all_probs),
    )

def main():
    args = parse_args()
    results_dir, checkpoint_dir = setup_directories(args)
    
    # Consistent suffix naming for files
    suffix = "_grayscale" if args.grayscale else "_color"
    
    checkpoint_name = args.checkpoint_name
    if checkpoint_name == "best.pt":
        checkpoint_name = f"best{suffix}.pt"
    checkpoint_path = checkpoint_dir / checkpoint_name
    
    # Automatically select best hardware accelerator
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")
    
    # 1. Dataset Preparation
    print("\n--- [Step 1/4] Preparing Dataset ---")
    dataset = PokemonSpriteDataset()
    train_idx, val_idx, test_idx = get_splits(args, dataset)
    
    # Save the splits definition inside the custom results directory for traceability
    split_info = {
        "split_strategy": args.split,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx
    }
    with open(results_dir / f"split_config{suffix}.json", "w") as f:
        json.dump(split_info, f)
    print(f"Saved split configuration metadata to: {results_dir / f'split_config{suffix}.json'}")
    
    if args.mode == "prepare":
        print("Dataset preparation complete. Exiting as requested.")
        return
        
    # 2. Model Training
    if args.mode in ["all", "train"]:
        print("\n--- [Step 2/4] Model Training ---")
        epochs = 2 if args.test_run else args.epochs
        print(f"Training for {epochs} epochs (freeze_backbone={args.freeze_backbone}, grayscale={args.grayscale})...")
        
        sampler = make_weighted_sampler(dataset, train_idx)
        train_loader = DataLoader(
            Subset(dataset, train_idx), 
            batch_size=args.batch_size, 
            sampler=sampler, 
            num_workers=0  # Safe and highly compatible
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx), 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        model = build_model(num_classes=len(TYPES), freeze_backbone=args.freeze_backbone).to(device)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), 
            lr=args.lr, 
            weight_decay=1e-2
        )
        criterion = nn.BCEWithLogitsLoss()
        
        best_f1, best_epoch = 0.0, 0
        csv_path = results_dir / f"training_log{suffix}.csv"
        csv_fields = ["epoch", "phase", "loss", "accuracy", "f1", "precision", "recall"]
        with open(csv_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=csv_fields).writeheader()
            
        active_train_tf = GRAYSCALE_TRAIN_TRANSFORM if args.grayscale else TRAIN_TRANSFORM
        active_eval_tf = GRAYSCALE_DEFAULT_TRANSFORM if args.grayscale else DEFAULT_TRANSFORM
            
        for epoch in range(1, epochs + 1):
            dataset.transform = active_train_tf
            train_m = run_epoch2(model, train_loader, criterion, optimizer, device, train=True)
            
            dataset.transform = active_eval_tf
            val_m   = run_epoch2(model, val_loader,   criterion, optimizer, device, train=False)
            
            log(epoch, epochs, "train", train_m)
            log(epoch, epochs, "val",   val_m)
            
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields)
                writer.writerow({"epoch": epoch, "phase": "train", **train_m})
                writer.writerow({"epoch": epoch, "phase": "val",   **val_m})
                
            if val_m["f1"] > best_f1:
                best_f1, best_epoch = val_m["f1"], epoch
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_f1": best_f1,
                    "args": vars(args),
                }, checkpoint_path)
                print(f"  ✓ Saved best checkpoint (val F1 {best_f1:.4f}) to: {checkpoint_path}")
                
        print(f"Training completed! Best val F1: {best_f1:.4f} at epoch {best_epoch}.")

    if args.mode == "train":
        return

    # 3. Evaluation
    if args.mode in ["all", "evaluate"]:
        print("\n--- [Step 3/4] Evaluating Model ---")
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found at {checkpoint_path}. Please train a model first.")
            sys.exit(1)
            
        print(f"Loading best checkpoint from: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model = build_model(num_classes=len(TYPES)).to(device)
        model.load_state_dict(ckpt["model_state"])
        
        # Ensure we use the correct transform for evaluation
        dataset.transform = GRAYSCALE_DEFAULT_TRANSFORM if args.grayscale else DEFAULT_TRANSFORM
        
        test_loader = DataLoader(
            Subset(dataset, test_idx), 
            batch_size=args.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        y_true, y_pred, y_probs = collect_predictions_pipeline(model, test_loader, device)
        
        # Save validation results locally in results dir
        np.save(results_dir / f"y_true{suffix}.npy", y_true)
        np.save(results_dir / f"y_pred{suffix}.npy", y_pred)
        np.save(results_dir / f"y_probs{suffix}.npy", y_probs)
        print(f"Saved y_true{suffix}.npy, y_pred{suffix}.npy, and y_probs{suffix}.npy to {results_dir}")
        
        print_summary(y_true, y_pred, y_probs)
        
        test_paths = [dataset.index[i][0] for i in test_idx]
        print_per_gen_metrics(y_true, y_pred, test_paths)

    # 4. Visualization
    if args.mode in ["all", "visualize"]:
        print("\n--- [Step 4/4] Generating Visualizations ---")
        y_true_path = results_dir / f"y_true{suffix}.npy"
        y_pred_path = results_dir / f"y_pred{suffix}.npy"
        y_probs_path = results_dir / f"y_probs{suffix}.npy"
        
        if not (y_true_path.exists() and y_pred_path.exists() and y_probs_path.exists()):
            print("Error: Evaluation outputs (npy files) not found. Run evaluation first.")
            sys.exit(1)
            
        y_true = np.load(y_true_path)
        y_pred = np.load(y_pred_path)
        y_probs = np.load(y_probs_path)
        
        test_paths = [dataset.index[i][0] for i in test_idx]
        
        # Ensure our visualization outputs go to our custom results directory!
        # Let's save both the simple mistakes gallery and the advanced detailed break downs.
        # We can dynamically monkeypatch evaluate's/Visualizer's RESULTS_DIR or implement our own clean versions.
        print("Generating diagnostic HTML mistake gallery...")
        
        # Build cards
        mistakes = []
        for i in range(len(y_true)):
            if not np.array_equal(y_true[i], y_pred[i]):
                true_types = [TYPES[j] for j in range(len(TYPES)) if y_true[i][j]]
                pred_types = [TYPES[j] for j in range(len(TYPES)) if y_pred[i][j]]
                partial = bool(np.logical_and(y_true[i], y_pred[i]).any())
                confidence = float(y_probs[i].max())
                mistakes.append((i, true_types, pred_types, confidence, partial))

        n_partial = sum(1 for *_, p in mistakes if p)
        n_wrong = len(mistakes) - n_partial

        # Sort: completely wrong first, then partial matches
        mistakes.sort(key=lambda x: (x[4], -x[3]))
        
        # CSS filter for grayscale if the flag was used
        img_style = "image-rendering:pixelated; filter: grayscale(100%);" if args.grayscale else "image-rendering:pixelated;"
        
        cards = []
        for i, true_types, pred_types, confidence, partial in mistakes:
            b64 = img_to_b64(test_paths[i])
            true_str = " / ".join(true_types)
            pred_str = " / ".join(pred_types) if pred_types else "(none)"
            border = "#ff4a9e" if partial else "#ff4b4b"
            tag = "Partial Match" if partial else "Completely Wrong"
            tag_color = "#ffbe5b" if partial else "#ff4b4b"

            type_prob_pairs = [(TYPES[j], float(y_probs[i][j]), bool(y_pred[i][j])) for j in range(len(TYPES))]
            type_prob_pairs.sort(key=lambda x: -x[1])

            prob_rows_html = ""
            for t_name, score, is_predicted in type_prob_pairs:
                if score >= 0.01:
                    highlight_style = "color:#ff4a9e; font-weight:bold;" if is_predicted else "color:#888;"
                    prob_rows_html += f"""
                    <div style="display:flex; justify-content:space-between; font-size:11px; margin:2px 0; {highlight_style}">
                      <span>{t_name}</span>
                      <span>{score:.1%}</span>
                    </div>
                    """

            cards.append(f"""
            <div style="display:inline-block; margin:10px; text-align:left; width:200px;
                        border:1px solid {border}; border-radius:12px; padding:12px; 
                        background:#13151a; box-shadow: 0 4px 20px rgba(0,0,0,0.4); vertical-align:top;
                        transition: transform 0.2s ease; cursor: pointer;">
              <div style="text-align:center; background:#1c1e24; border-radius:8px; padding:8px; display: flex; justify-content: center; align-items: center;">
                <img src="data:image/png;base64,{b64}" width="96" height="96" style="{img_style}"/><br>
              </div>
              <div style="margin-top:10px; border-bottom:1px solid #222530; padding-bottom:8px; margin-bottom:8px;">
                <span style="color:#00a3ff; font-size:13px; font-weight:bold; display:block; margin-bottom: 2px;">True: {true_str}</span>
                <span style="color:#ff4b4b; font-size:13px; display:block; margin-bottom: 4px;">Pred: {pred_str}</span>
                <span style="color:{tag_color}; font-size:11px; font-weight:bold; display:block;">{tag}</span>
              </div>
              <div style="background:#0d0e12; padding:8px; border-radius:6px;">
                <div style="font-size:9px; color:#555; text-transform:uppercase; font-weight:bold; margin-bottom:6px; border-bottom:1px solid #1c1e24; padding-bottom: 2px;">Model Confidences</div>
                {prob_rows_html}
              </div>
            </div>""")

        total = len(y_true)
        html = f"""<!DOCTYPE html>
        <html>
        <head>
          <meta charset="utf-8">
          <title>Pokemon Classification - Mistakes Gallery</title>
          <style>
            body {{
              background: #090a0f;
              color: #f1f1f1;
              font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
              padding: 24px;
              margin: 0;
            }}
            .container {{
              max-width: 1200px;
              margin: 0 auto;
            }}
            .header {{
              border-bottom: 1px solid #222530;
              padding-bottom: 20px;
              margin-bottom: 24px;
            }}
            .stats {{
              display: flex;
              gap: 20px;
              margin-top: 10px;
            }}
            .stat-badge {{
              padding: 6px 12px;
              border-radius: 20px;
              font-size: 13px;
              font-weight: 600;
            }}
            .gallery {{
              display: flex;
              flex-wrap: wrap;
              justify-content: flex-start;
            }}
          </style>
        </head>
        <body>
          <div class="container">
            <div class="header">
              <h1 style="margin:0; font-size:28px; background: linear-gradient(135deg, #00a3ff, #ff4a9e); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                CNN Classification Mistake Diagnostic Breakdown
              </h1>
              <div class="stats">
                <span class="stat-badge" style="background: rgba(255,255,255,0.05); color: #fff;">Total Tested: {total}</span>
                <span class="stat-badge" style="background: rgba(255,75,75,0.15); color: #ff4b4b;">Total Mistakes: {len(mistakes)}</span>
                <span class="stat-badge" style="background: rgba(255,190,91,0.15); color: #ffbe5b;">Partial Matches: {n_partial}</span>
                <span class="stat-badge" style="background: rgba(255,75,75,0.25); color: #ff4b4b;">Completely Wrong: {n_wrong}</span>
              </div>
            </div>
            <div class="gallery">
              {"".join(cards)}
            </div>
          </div>
        </body>
        </html>"""

        gallery_out = results_dir / f"mistakes_{'grayscale' if args.grayscale else 'color'}_gallery.html"
        gallery_out.write_text(html, encoding="utf-8")
        print(f"Successfully generated custom premium mistake gallery ({'grayscale' if args.grayscale else 'color'}) at: {gallery_out}")
        print("\nAll pipeline tasks successfully completed!")

if __name__ == "__main__":
    main()