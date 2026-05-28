#!/usr/bin/env python3
"""Unified train/evaluate/visualize pipeline for Pokemon type classification."""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

classification_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(classification_dir))

from cnn_model import build_model
from dataset import (
    DEFAULT_TRANSFORM,
    GRAYSCALE_DEFAULT_TRANSFORM,
    GRAYSCALE_TRAIN_TRANSFORM,
    TRAIN_TRANSFORM,
    PokemonSpriteDataset,
    TYPES,
    gen_gen_split,
    gen_stratified_split,
)
from evaluate import img_to_b64, print_per_gen_metrics, print_summary
from prediction import GAP_THRESHOLD, collect_predictions
from train import log, make_weighted_sampler, run_epoch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified pipeline for Pokemon type classification."
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "prepare", "train", "evaluate", "visualize"],
        help="Pipeline step to run.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="stratified",
        choices=["stratified", "generation"],
        help="Split strategy.",
    )
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--test-frac", type=float, default=0.15)
    parser.add_argument("--train-gens", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--test-gens", type=int, nargs="+", default=[4, 5, 6])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--test-run", action="store_true", help="Train for 2 epochs.")
    parser.add_argument("--grayscale", action="store_true", help="Use grayscale sprite inputs.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker count.")
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--checkpoint-name", type=str, default="best.pt")
    return parser.parse_args()


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_directories(args):
    results_dir = Path(args.results_dir) if args.results_dir else classification_dir / "results"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else classification_dir / "checkpoints"
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return results_dir, checkpoint_dir


def get_splits(args, dataset):
    if args.split == "stratified":
        print(
            "Generating stratified split "
            f"(val_frac={args.val_frac}, test_frac={args.test_frac}, seed={args.seed})..."
        )
        train_idx, val_idx, test_idx = gen_stratified_split(
            dataset.index,
            val_frac=args.val_frac,
            test_frac=args.test_frac,
            seed=args.seed,
        )
    else:
        print(
            "Generating generation split "
            f"(train_gens={args.train_gens}, test_gens={args.test_gens}, "
            f"val_frac={args.val_frac}, seed={args.seed})..."
        )
        train_idx, val_idx, test_idx = gen_gen_split(
            dataset.index,
            train_gens=args.train_gens,
            val_frac=args.val_frac,
            test_gens=args.test_gens,
            seed=args.seed,
        )

    print(f"Split sizes -> Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
    return train_idx, val_idx, test_idx


def save_split_config(args, train_idx, val_idx, test_idx, results_dir):
    split_info = {
        "split_strategy": args.split,
        "train_indices": train_idx,
        "val_indices": val_idx,
        "test_indices": test_idx,
    }
    out = results_dir / "split_config.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(split_info, f)
    print(f"Saved split metadata to: {out}")


def train_model(args, dataset, train_idx, val_idx, device, checkpoint_path, results_dir):
    epochs = 2 if args.test_run else args.epochs
    print(
        f"Training for {epochs} epochs "
        f"(freeze_backbone={args.freeze_backbone}, grayscale={args.grayscale})..."
    )

    sampler = make_weighted_sampler(dataset, train_idx)
    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = build_model(num_classes=len(TYPES), freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-2,
    )
    criterion = nn.BCEWithLogitsLoss()

    csv_path = results_dir / "training_log.csv"
    csv_fields = ["epoch", "phase", "loss", "accuracy", "f1", "precision", "recall"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    active_train_tf = GRAYSCALE_TRAIN_TRANSFORM if args.grayscale else TRAIN_TRANSFORM
    active_eval_tf = GRAYSCALE_DEFAULT_TRANSFORM if args.grayscale else DEFAULT_TRANSFORM
    best_f1, best_epoch = 0.0, 0

    print(f"Prediction rule: top type plus second type when probability gap < {GAP_THRESHOLD}")
    for epoch in range(1, epochs + 1):
        dataset.transform = active_train_tf
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        dataset.transform = active_eval_tf
        val_m = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        log(epoch, epochs, "train", train_m)
        log(epoch, epochs, "val", val_m)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({"epoch": epoch, "phase": "train", **train_m})
            writer.writerow({"epoch": epoch, "phase": "val", **val_m})

        if val_m["f1"] > best_f1:
            best_f1, best_epoch = val_m["f1"], epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_f1": best_f1,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  Saved best checkpoint (val F1 {best_f1:.4f}) to: {checkpoint_path}")

    print(f"Training complete. Best val F1: {best_f1:.4f} at epoch {best_epoch}.")


def evaluate_model(args, dataset, test_idx, device, checkpoint_path, results_dir):
    if not checkpoint_path.exists():
        print(f"Error: checkpoint not found at {checkpoint_path}. Train a model first.")
        sys.exit(1)

    print(f"Loading best checkpoint from: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(num_classes=len(TYPES)).to(device)
    model.load_state_dict(ckpt["model_state"])
    dataset.transform = GRAYSCALE_DEFAULT_TRANSFORM if args.grayscale else DEFAULT_TRANSFORM

    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    y_true, y_pred, y_probs = collect_predictions(model, test_loader, device)

    np.save(results_dir / "y_true.npy", y_true)
    np.save(results_dir / "y_pred.npy", y_pred)
    np.save(results_dir / "y_probs.npy", y_probs)
    print(f"Saved y_true.npy, y_pred.npy, and y_probs.npy to {results_dir}")

    print_summary(y_true, y_pred, y_probs)
    test_paths = [dataset.index[i][0] for i in test_idx]
    print_per_gen_metrics(y_true, y_pred, test_paths)


def save_pipeline_gallery(args, dataset, test_idx, results_dir):
    y_true_path = results_dir / "y_true.npy"
    y_pred_path = results_dir / "y_pred.npy"
    y_probs_path = results_dir / "y_probs.npy"
    if not (y_true_path.exists() and y_pred_path.exists() and y_probs_path.exists()):
        print("Error: evaluation outputs not found. Run evaluation first.")
        sys.exit(1)

    y_true = np.load(y_true_path)
    y_pred = np.load(y_pred_path)
    y_probs = np.load(y_probs_path)
    test_paths = [dataset.index[i][0] for i in test_idx]

    mistakes = []
    for i in range(len(y_true)):
        if not np.array_equal(y_true[i], y_pred[i]):
            true_types = [TYPES[j] for j in range(len(TYPES)) if y_true[i][j]]
            pred_types = [TYPES[j] for j in range(len(TYPES)) if y_pred[i][j]]
            partial = bool(np.logical_and(y_true[i], y_pred[i]).any())
            confidence = float(y_probs[i].max())
            mistakes.append((i, true_types, pred_types, confidence, partial))

    n_partial = sum(1 for *_, partial in mistakes if partial)
    n_wrong = len(mistakes) - n_partial
    mistakes.sort(key=lambda x: (x[4], -x[3]))
    img_style = (
        "image-rendering:pixelated; filter: grayscale(100%);"
        if args.grayscale
        else "image-rendering:pixelated;"
    )

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
        for type_name, score, is_predicted in type_prob_pairs:
            if score >= 0.01:
                highlight_style = "color:#ff4a9e; font-weight:bold;" if is_predicted else "color:#888;"
                prob_rows_html += f"""
                    <div style="display:flex; justify-content:space-between; font-size:11px; margin:2px 0; {highlight_style}">
                      <span>{type_name}</span>
                      <span>{score:.1%}</span>
                    </div>
                    """

        cards.append(
            f"""
            <div style="display:inline-block; margin:10px; text-align:left; width:200px;
                        border:1px solid {border}; border-radius:12px; padding:12px;
                        background:#13151a; box-shadow: 0 4px 20px rgba(0,0,0,0.4); vertical-align:top;">
              <div style="text-align:center; background:#1c1e24; border-radius:8px; padding:8px; display:flex; justify-content:center; align-items:center;">
                <img src="data:image/png;base64,{b64}" width="96" height="96" style="{img_style}"/>
              </div>
              <div style="margin-top:10px; border-bottom:1px solid #222530; padding-bottom:8px; margin-bottom:8px;">
                <span style="color:#00a3ff; font-size:13px; font-weight:bold; display:block; margin-bottom:2px;">True: {true_str}</span>
                <span style="color:#ff4b4b; font-size:13px; display:block; margin-bottom:4px;">Pred: {pred_str}</span>
                <span style="color:{tag_color}; font-size:11px; font-weight:bold; display:block;">{tag} | Conf: {confidence:.1%}</span>
              </div>
              <div style="background:#0d0e12; padding:8px; border-radius:6px;">
                <div style="font-size:9px; color:#555; text-transform:uppercase; font-weight:bold; margin-bottom:6px; border-bottom:1px solid #1c1e24; padding-bottom:2px;">Model Confidences</div>
                {prob_rows_html}
              </div>
            </div>"""
        )

    total = len(y_true)
    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pokemon Classification - Mistakes Gallery</title>
  <style>
    body {{
      background:#090a0f;
      color:#f1f1f1;
      font-family:Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      padding:24px;
      margin:0;
    }}
    .container {{ max-width:1200px; margin:0 auto; }}
    .header {{ border-bottom:1px solid #222530; padding-bottom:20px; margin-bottom:24px; }}
    .stats {{ display:flex; gap:20px; margin-top:10px; flex-wrap:wrap; }}
    .stat-badge {{ padding:6px 12px; border-radius:20px; font-size:13px; font-weight:600; }}
    .gallery {{ display:flex; flex-wrap:wrap; justify-content:flex-start; }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1 style="margin:0; font-size:28px;">CNN Classification Mistake Diagnostic Breakdown</h1>
      <div class="stats">
        <span class="stat-badge" style="background:rgba(255,255,255,0.05); color:#fff;">Total Tested: {total}</span>
        <span class="stat-badge" style="background:rgba(255,75,75,0.15); color:#ff4b4b;">Total Mistakes: {len(mistakes)}</span>
        <span class="stat-badge" style="background:rgba(255,190,91,0.15); color:#ffbe5b;">Partial Matches: {n_partial}</span>
        <span class="stat-badge" style="background:rgba(255,75,75,0.25); color:#ff4b4b;">Completely Wrong: {n_wrong}</span>
      </div>
    </div>
    <div class="gallery">{"".join(cards)}</div>
  </div>
</body>
</html>"""

    gallery_out = results_dir / f"mistakes_{'grayscale' if args.grayscale else 'color'}_gallery.html"
    gallery_out.write_text(html, encoding="utf-8")
    print(f"Saved gallery to: {gallery_out}")


def main():
    args = parse_args()
    results_dir, checkpoint_dir = setup_directories(args)
    checkpoint_path = checkpoint_dir / args.checkpoint_name

    device = select_device()
    print(f"Device: {device}")

    print("\n--- [Step 1/4] Preparing Dataset ---")
    dataset = PokemonSpriteDataset()
    train_idx, val_idx, test_idx = get_splits(args, dataset)
    save_split_config(args, train_idx, val_idx, test_idx, results_dir)

    if args.mode == "prepare":
        print("Dataset preparation complete.")
        return

    if args.mode in ["all", "train"]:
        print("\n--- [Step 2/4] Model Training ---")
        train_model(args, dataset, train_idx, val_idx, device, checkpoint_path, results_dir)

    if args.mode == "train":
        return

    if args.mode in ["all", "evaluate"]:
        print("\n--- [Step 3/4] Evaluating Model ---")
        evaluate_model(args, dataset, test_idx, device, checkpoint_path, results_dir)

    if args.mode in ["all", "visualize"]:
        print("\n--- [Step 4/4] Generating Visualizations ---")
        save_pipeline_gallery(args, dataset, test_idx, results_dir)
        print("\nAll pipeline tasks completed.")


if __name__ == "__main__":
    main()
