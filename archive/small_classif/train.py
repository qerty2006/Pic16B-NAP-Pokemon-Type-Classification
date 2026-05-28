import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm


sys.path.insert(0, str(Path(__file__).parent))
from dataset import gen_stratified_split
from ViT import build_vit_b16
from dataset import PokemonSpriteDataset, TYPES, gen_gen_split, TRAIN_TRANSFORM, DEFAULT_TRANSFORM
from cnn_model import build_model

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def make_weighted_sampler(dataset, train_idx):
    # 1. Calculate class counts across the whole dataset
    # (or just the training set, depending on preference)
    labels = np.array([dataset.index[i][1] for i in train_idx])
    class_counts = labels.sum(axis=0)
    class_counts = np.where(class_counts == 0, 1, class_counts)

    weights = []
    # 2. Loop directly through the absolute dataset indices!
    for idx in train_idx:
        # Look up the actual labels using the real dataset position
        sample_labels = dataset.index[idx][1]

        if sample_labels.any():
            # Find the rarest type this specific Pokémon possesses
            weight = (1.0 / class_counts[sample_labels.astype(bool)]).max()
        else:
            weight = 1.0
        weights.append(weight)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Run one full pass over loader in either train or eval mode.

    In train mode: runs backprop and updates weights each batch.
    In eval mode: no gradients, no weight updates.

    Returns a dict with keys: loss, accuracy (exact match), f1 (macro),
    precision (macro), recall (macro). Metrics are computed using PRED_THRESHOLD
    on sigmoid outputs — this is for display during training only; see evaluate.py
    for the top-k evaluation used at test time.
    """
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    phase = "train" if train else "val"
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc=phase, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)
            # Threshold-based preds for live training display only; evaluate.py uses top-k instead
            preds = (torch.sigmoid(logits) > 0.5).cpu().int().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().int().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    n = len(all_labels)
    metrics = {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),   # exact match
        "f1":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    return metrics



def run_epoch2(model, loader, criterion, optimizer, device, train=True):
    """Run one full pass over loader in either train or eval mode.

    In train mode: runs backprop and updates weights each batch.
    In eval mode: no gradients, no weight updates.

    Returns a dict with keys: loss, accuracy (exact match), f1 (macro),
    precision (macro), recall (macro). Metrics are computed using PRED_THRESHOLD
    on sigmoid outputs — this is for display during training only; see evaluate.py
    for the top-k evaluation used at test time.
    """
    model.train() if train else model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    phase = "train" if train else "val"
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in tqdm(loader, desc=phase, leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(labels)

            # === ADD THESE LINES INSIDE THE FOR LOOP ===
            probs = torch.sigmoid(logits)
            sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
            batch_preds = torch.zeros_like(probs, dtype=torch.int)

            GAP_THRESHOLD = 0.25  # Adjust this value to tune single vs double types

            for i in range(len(labels)):
                # Always predict the #1 top confidence type
                batch_preds[i, sorted_idx[i, 0]] = 1

                # Check if the 2nd type is close enough to the 1st
                if (sorted_probs[i, 0] - sorted_probs[i, 1]) < GAP_THRESHOLD:
                    batch_preds[i, sorted_idx[i, 1]] = 1

            all_preds.append(batch_preds)
            all_labels.append(labels.cpu().int().numpy())
            # ===========================================

    all_preds = torch.cat(all_preds).cpu().numpy()
    all_labels = np.vstack(all_labels)
    n = len(all_labels)
    metrics = {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),   # exact match
        "f1":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }
    return metrics

def log(epoch, total_epochs, phase, m):
    """Print one epoch's metrics to stdout in a fixed-width format."""
    print(
        f"[{epoch:>3}/{total_epochs}] {phase:<5} | "
        f"loss {m['loss']:.4f} | acc {m['accuracy']:.4f} | "
        f"f1 {m['f1']:.4f} | prec {m['precision']:.4f} | rec {m['recall']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true",
                        help="Only train the classifier head")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    print("Building dataset...")



    # ==== CAN REDUCE ====

    """ BELOW IS SECTION FOR GEN GEN    
    #base_dataset = PokemonSpriteDataset()
    #train_idx, val_idx, _ = gen_gen_split(base_dataset.index, train_gens=(1, 2, 3), test_gens=(4, 5, 6))
    #print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}")

    #train_dataset = PokemonSpriteDataset(transform=TRAIN_TRANSFORM)
    #val_dataset = PokemonSpriteDataset(transform=DEFAULT_TRANSFORM)

    #sampler = make_weighted_sampler(train_dataset, train_idx)
    # num_workers=0 loads data in the main process — safe on Windows, lower RAM usage.
    # Increase to 2-4 on Linux/Mac or if you have spare RAM for faster data loading. """

    """train_loader = DataLoader(
           Subset(train_dataset, train_idx), batch_size=args.batch_size, sampler=sampler, num_workers=4
       )
       val_loader = DataLoader(
           Subset(val_dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=4
       )"""

    dataset = PokemonSpriteDataset()
    train_idx, val_idx, _ = gen_stratified_split(dataset.index)
    sampler = make_weighted_sampler(dataset, train_idx)

    train_loader = DataLoader(
    Subset(dataset, train_idx), batch_size = args.batch_size, sampler = sampler, num_workers = 4)

    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=4
    )


     # ==== CAN REDUCE ====


    #model = build_vit_b16(num_classes=len(TYPES), freeze_backbone=True).to(device)
    model = build_model(num_classes=len(TYPES), freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2
    )


    # patience=5: halves LR if val loss doesn't improve for 5 consecutive epochs
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_f1, best_epoch = 0.0, 0

    # log.csv is NOT gitignored — back it up before pulling; teammates' runs will overwrite it
    csv_path = Path(__file__).parent / "log.csv"
    csv_fields = ["epoch", "phase", "loss", "accuracy", "f1", "precision", "recall"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    print(f"SUCCESS: Using model architecture -> {type(model).__name__}")
    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        dataset.transform = TRAIN_TRANSFORM
        train_m = run_epoch2(model, train_loader, criterion, optimizer, device, train=True)
        dataset.transform = DEFAULT_TRANSFORM
        val_m   = run_epoch2(model, val_loader,   criterion, optimizer, device, train=False)

        log(epoch, args.epochs, "train", train_m)
        log(epoch, args.epochs, "val",   val_m)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({"epoch": epoch, "phase": "train", **train_m})
            writer.writerow({"epoch": epoch, "phase": "val",   **val_m})

        #scheduler.step(val_m["loss"])

        if val_m["f1"] > best_f1:
            best_f1, best_epoch = val_m["f1"], epoch
            # best.pt IS gitignored — copy it elsewhere before switching branches or pulling
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_f1": best_f1,
                "args": vars(args),
            }, CHECKPOINT_DIR / "best.pt")
            print(f"  ✓ Saved best checkpoint (val F1 {best_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
