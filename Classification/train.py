import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from dataset import PokemonSpriteDataset, TYPES, gen_stratified_split
from cnn_model import build_resnet18

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def make_weighted_sampler(dataset, train_idx):
    labels = np.array([dataset.index[i][1] for i in train_idx])
    class_counts = np.bincount(labels, minlength=len(TYPES))
    class_counts = np.where(class_counts == 0, 1, class_counts)
    weights = 1.0 / class_counts[labels]
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
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    metrics = {
        "loss":      total_loss / n,
        "accuracy":  accuracy_score(all_labels, all_preds),
        "f1":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "recall":    recall_score(all_labels, all_preds, average="macro", zero_division=0),
    }
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("Building dataset...")
    dataset = PokemonSpriteDataset()
    train_idx, val_idx, _ = gen_stratified_split(dataset.index)
    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}")

    sampler = make_weighted_sampler(dataset, train_idx)
    train_loader = DataLoader(
        Subset(dataset, train_idx), batch_size=args.batch_size, sampler=sampler, num_workers=2
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx), batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    model = build_resnet18(num_classes=len(TYPES), freeze_backbone=args.freeze_backbone).to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = nn.CrossEntropyLoss()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    best_f1, best_epoch = 0.0, 0

    for epoch in tqdm(range(1, args.epochs + 1), desc="Epochs"):
        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)

        log(epoch, args.epochs, "train", train_m)
        log(epoch, args.epochs, "val",   val_m)

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
