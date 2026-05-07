import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent))
from dataset import PokemonSpriteDataset, TYPES, gen_stratified_split
from cnn_model import build_resnet18

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best.pt"
RESULTS_DIR = Path(__file__).parent / "results"


def collect_predictions(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = F.softmax(logits, dim=1)
            all_labels.extend(labels.tolist())
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_probs.append(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.vstack(all_probs),
    )


def print_summary(y_true, y_pred, y_probs):
    acc  = accuracy_score(y_true, y_pred)
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = float("nan")

    print("\n=== Overall Metrics ===")
    print(f"  Accuracy:       {acc:.4f}")
    print(f"  F1 (macro):     {f1:.4f}")
    print(f"  Precision:      {prec:.4f}")
    print(f"  Recall:         {rec:.4f}")
    print(f"  ROC-AUC (ovr):  {auc:.4f}")

    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(TYPES))))
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(TYPES))))
    rec_per  = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(TYPES))))

    print("\n=== Per-Type Metrics ===")
    print(f"  {'Type':<12} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*33}")
    for i, t in enumerate(TYPES):
        print(f"  {t:<12} {f1_per[i]:>6.4f} {prec_per[i]:>6.4f} {rec_per[i]:>6.4f}")


def print_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(TYPES))))
    print("\n=== Confusion Matrix (rows=true, cols=predicted) ===")
    header = f"{'':>10}" + "".join(f"{t[:4]:>6}" for t in TYPES)
    print(header)
    for i, t in enumerate(TYPES):
        row = f"{t:<10}" + "".join(f"{cm[i, j]:>6}" for j in range(len(TYPES)))
        print(row)
    return cm


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Run train.py first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val F1: {ckpt['val_f1']:.4f})")

    model = build_resnet18(num_classes=len(TYPES)).to(device)
    model.load_state_dict(ckpt["model_state"])

    dataset = PokemonSpriteDataset()
    _, _, test_idx = gen_stratified_split(dataset.index)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False, num_workers=2)
    print(f"Test set size: {len(test_idx)}")

    y_true, y_pred, y_probs = collect_predictions(model, test_loader, device)

    print_summary(y_true, y_pred, y_probs)
    cm = print_confusion_matrix(y_true, y_pred)

    # save outputs for Ajmain's visualizations
    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / "confusion_matrix.npy", cm)
    np.save(RESULTS_DIR / "y_true.npy", y_true)
    np.save(RESULTS_DIR / "y_pred.npy", y_pred)
    np.save(RESULTS_DIR / "y_probs.npy", y_probs)
    print(f"\nResults saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
