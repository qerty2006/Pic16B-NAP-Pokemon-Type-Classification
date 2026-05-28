import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from dataset import PokemonSpriteDataset, TYPES, gen_gen_split, TRAIN_TRANSFORM, DEFAULT_TRANSFORM
from cnn_model import build_model
from prediction_utils import multilabel_metrics, predict_from_probabilities

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def make_weighted_sampler(dataset, train_idx):
    labels = np.array([dataset.index[i][1] for i in train_idx])  # (N, 18) multi-hot
    class_counts = labels.sum(axis=0)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    # weight each sample by its rarest type to counteract class imbalance
    weights = np.array([
        (1.0 / class_counts[labels[i].astype(bool)]).max() if labels[i].any() else 1.0
        for i in range(len(labels))
    ])
    return WeightedRandomSampler(weights.tolist(), num_samples=len(weights), replacement=True)


def run_epoch(model, loader, criterion, optimizer, device, train=True):
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
            preds = predict_from_probabilities(torch.sigmoid(logits)).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.cpu().int().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    metrics = multilabel_metrics(all_labels, all_preds)
    metrics["loss"] = total_loss / len(all_labels)
    return metrics


def log(epoch, total_epochs, phase, m):
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
    base_dataset = PokemonSpriteDataset()
    train_idx, val_idx, _ = gen_gen_split(base_dataset.index, train_gens=(1, 2, 3), test_gens=(4, 5, 6))
    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}")

    train_dataset = PokemonSpriteDataset(transform=TRAIN_TRANSFORM)
    val_dataset = PokemonSpriteDataset(transform=DEFAULT_TRANSFORM)

    sampler = make_weighted_sampler(train_dataset, train_idx)
    # num_workers=0 loads data in the main process — safe on Windows, lower RAM usage.
    # Increase to 2-4 on Linux/Mac or if you have spare RAM for faster data loading.
    train_loader = DataLoader(
        Subset(train_dataset, train_idx), batch_size=args.batch_size, sampler=sampler, num_workers=4
    )
    val_loader = DataLoader(
        Subset(val_dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    model = build_model(num_classes=len(TYPES), freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-2
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.BCEWithLogitsLoss()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_f1, best_epoch = 0.0, 0

    csv_path = Path(__file__).parent / "log.csv"
    csv_fields = ["epoch", "phase", "loss", "accuracy", "f1", "precision", "recall"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        log(epoch, args.epochs, "train", train_m)
        log(epoch, args.epochs, "val",   val_m)

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow({"epoch": epoch, "phase": "train", **train_m})
            writer.writerow({"epoch": epoch, "phase": "val",   **val_m})

        scheduler.step(val_m["loss"])

        if val_m["f1"] > best_f1:
            best_f1, best_epoch = val_m["f1"], epoch
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
