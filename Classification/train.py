import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
<<<<<<< Updated upstream
from dataset import PokemonSpriteDataset, TYPES, gen_stratified_split, PRED_THRESHOLD
from cnn_model import build_efficientnet_b0
=======
from cnn_model import build_model
from dataset import DEFAULT_TRANSFORM, TRAIN_TRANSFORM, PokemonSpriteDataset, TYPES, gen_stratified_split
from prediction_utils import multilabel_metrics, predict_from_probabilities
>>>>>>> Stashed changes

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def make_weighted_sampler(dataset, train_idx):
<<<<<<< Updated upstream
    """Build a WeightedRandomSampler that upsamples training examples containing rare types.

    Each sample's weight is the inverse frequency of its rarest type, so the sampler
    draws rare-type Pokemon more often. Without this, the model collapses to predicting
    Water/Normal for everything because they dominate the dataset.

    Returns a WeightedRandomSampler with replacement, same length as train_idx.
    """
    labels = np.array([dataset.index[i][1] for i in train_idx])  # (N, 18) multi-hot
    class_counts = labels.sum(axis=0)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    # weight each sample by its rarest type to counteract class imbalance
    weights = np.array([
        (1.0 / class_counts[labels[i].astype(bool)]).max() if labels[i].any() else 1.0
        for i in range(len(labels))
    ])
    return WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)
=======
    labels = np.array([dataset.index[i][1] for i in train_idx])
    class_counts = labels.sum(axis=0)
    class_counts = np.where(class_counts == 0, 1, class_counts)

    weights = []
    for idx in train_idx:
        sample_labels = dataset.index[idx][1]
        if sample_labels.any():
            # Weight each sample by its rarest type to counter class imbalance.
            weight = (1.0 / class_counts[sample_labels.astype(bool)]).max()
        else:
            weight = 1.0
        weights.append(weight)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
>>>>>>> Stashed changes


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """Run one training or validation pass using the shared gap prediction rule."""
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
<<<<<<< Updated upstream
            # Threshold-based preds for live training display only; evaluate.py uses top-k instead
            preds = (torch.sigmoid(logits) > PRED_THRESHOLD).cpu().int().numpy()
=======
            probs = torch.sigmoid(logits)
            preds = predict_from_probabilities(probs).cpu().numpy()
>>>>>>> Stashed changes
            all_preds.append(preds)
            all_labels.append(labels.cpu().int().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = multilabel_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


<<<<<<< Updated upstream
def log(epoch, total_epochs, phase, m):
    """Print one epoch's metrics to stdout in a fixed-width format."""
=======
def log(epoch, total_epochs, phase, metrics):
>>>>>>> Stashed changes
    print(
        f"[{epoch:>3}/{total_epochs}] {phase:<5} | "
        f"loss {metrics['loss']:.4f} | acc {metrics['accuracy']:.4f} | "
        f"f1 {metrics['f1']:.4f} | prec {metrics['precision']:.4f} | rec {metrics['recall']:.4f}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train the classifier head")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building dataset...")
    dataset = PokemonSpriteDataset()
    train_idx, val_idx, _ = gen_stratified_split(dataset.index)
    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}")

    sampler = make_weighted_sampler(dataset, train_idx)
    # num_workers=0 loads data in the main process — safe on Windows, lower RAM usage.
    # Increase to 2-4 on Linux/Mac or if you have spare RAM for faster data loading.
    train_loader = DataLoader(
<<<<<<< Updated upstream
        Subset(dataset, train_idx), batch_size=args.batch_size, sampler=sampler, num_workers=0
=======
        Subset(dataset, train_idx), batch_size=args.batch_size, sampler=sampler, num_workers=4
>>>>>>> Stashed changes
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=0
    )

<<<<<<< Updated upstream
    model = build_efficientnet_b0(num_classes=len(TYPES), freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    # patience=5: halves LR if val loss doesn't improve for 5 consecutive epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
=======
    model = build_model(num_classes=len(TYPES), freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2
    )
>>>>>>> Stashed changes
    criterion = nn.BCEWithLogitsLoss()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_f1, best_epoch = 0.0, 0

    csv_path = Path(__file__).parent / "log.csv"
    csv_fields = ["epoch", "phase", "loss", "accuracy", "f1", "precision", "recall"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
<<<<<<< Updated upstream
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
=======
        dataset.transform = TRAIN_TRANSFORM
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        dataset.transform = DEFAULT_TRANSFORM
        val_m = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
>>>>>>> Stashed changes

        log(epoch, args.epochs, "train", train_m)
        log(epoch, args.epochs, "val", val_m)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({"epoch": epoch, "phase": "train", **train_m})
<<<<<<< Updated upstream
            writer.writerow({"epoch": epoch, "phase": "val",   **val_m})

        scheduler.step(val_m["loss"])
=======
            writer.writerow({"epoch": epoch, "phase": "val", **val_m})
>>>>>>> Stashed changes

        if val_m["f1"] > best_f1:
            best_f1, best_epoch = val_m["f1"], epoch
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "val_f1": best_f1,
                "args": vars(args),
            }, CHECKPOINT_DIR / "best.pt")
            print(f"  Saved best checkpoint (val F1 {best_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f} at epoch {best_epoch}.")


if __name__ == "__main__":
    main()
