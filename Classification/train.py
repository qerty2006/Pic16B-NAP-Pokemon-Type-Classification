import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from cnn_model import build_model
from dataset import DEFAULT_TRANSFORM, PokemonSpriteDataset, TRAIN_TRANSFORM, TYPES, gen_stratified_split
from prediction import GAP_THRESHOLD, predict_from_probs

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"
RESULTS_DIR = Path(__file__).parent / "results"


def make_weighted_sampler(dataset, train_idx):
    """Weight samples by the rarest type they contain."""
    labels = np.array([dataset.index[i][1] for i in train_idx])
    class_counts = labels.sum(axis=0)
    class_counts = np.where(class_counts == 0, 1, class_counts)

    weights = []
    for idx in train_idx:
        sample_labels = dataset.index[idx][1]
        if sample_labels.any():
            weight = (1.0 / class_counts[sample_labels.astype(bool)]).max()
        else:
            weight = 1.0
        weights.append(weight)

    return WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)


def run_epoch(model, loader, criterion, optimizer, device, train=True, gap_threshold=GAP_THRESHOLD):
    """Run one training or validation pass and compute gap-threshold metrics."""
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
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.append(predict_from_probs(probs, gap_threshold=gap_threshold))
            all_labels.append(labels.detach().cpu().int().numpy())

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    n = len(all_labels)
    return {
        "loss": total_loss / n,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }


def log(epoch, total_epochs, phase, metrics):
    """Print one epoch's metrics to stdout in a fixed-width format."""
    print(
        f"[{epoch:>3}/{total_epochs}] {phase:<5} | "
        f"loss {metrics['loss']:.4f} | acc {metrics['accuracy']:.4f} | "
        f"f1 {metrics['f1']:.4f} | prec {metrics['precision']:.4f} | rec {metrics['recall']:.4f}"
    )


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the EfficientNet-B0 Pokemon type classifier.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true", help="Only train the classifier head")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count")
    return parser.parse_args()


def main():
    args = parse_args()
    device = select_device()
    print(f"Device: {device}")

    print("Building dataset...")
    dataset = PokemonSpriteDataset()
    train_idx, val_idx, _ = gen_stratified_split(dataset.index)
    print(f"Split - train: {len(train_idx)}, val: {len(val_idx)}")

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

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "training_log.csv"
    csv_fields = ["epoch", "phase", "loss", "accuracy", "f1", "precision", "recall"]
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=csv_fields).writeheader()

    best_f1, best_epoch = 0.0, 0
    print(f"Using model architecture: {type(model).__name__}")
    print(f"Prediction rule: top type plus second type when probability gap < {GAP_THRESHOLD}")

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        dataset.transform = TRAIN_TRANSFORM
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        dataset.transform = DEFAULT_TRANSFORM
        val_m = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        log(epoch, args.epochs, "train", train_m)
        log(epoch, args.epochs, "val", val_m)

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
                CHECKPOINT_DIR / "best.pt",
            )
            print(f"  Saved best checkpoint (val F1 {best_f1:.4f})")

    print(f"\nTraining complete. Best val F1: {best_f1:.4f} at epoch {best_epoch}.")
    print(f"Training log saved to {csv_path}")


if __name__ == "__main__":
    main()
