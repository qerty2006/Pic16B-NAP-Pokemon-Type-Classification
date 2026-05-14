import sys
import base64
import io
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))
from dataset import PokemonSpriteDataset, TYPES, gen_gen_split, get_generation, PRED_THRESHOLD
import generate_report
from cnn_model import build_model

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "best.pt"
RESULTS_DIR = Path(__file__).parent / "results"


def collect_predictions(model, loader, device, n_types):
    """Predict exactly as many types as each Pokemon actually has (top-k per sample)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    sample_idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            batch_size = len(labels)

            preds = np.zeros_like(probs, dtype=int)
            for i in range(batch_size):
                k = n_types[sample_idx + i]
                top_k_idx = np.argsort(probs[i])[-k:]
                preds[i, top_k_idx] = 1

            all_labels.append(labels.int().numpy())
            all_preds.append(preds)
            all_probs.append(probs)
            sample_idx += batch_size

    return (
        np.vstack(all_labels),
        np.vstack(all_preds),
        np.vstack(all_probs),
    )


def print_summary(y_true, y_pred, y_probs):
    acc  = accuracy_score(y_true, y_pred)   # exact match across both types
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs, average="macro")
    except ValueError:
        auc = float("nan")

    print("\n=== Overall Metrics ===")
    print(f"  Exact Match Acc:  {acc:.4f}")
    print(f"  F1 (macro):       {f1:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  ROC-AUC:          {auc:.4f}")

    f1_per   = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(TYPES))))
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(TYPES))))
    rec_per  = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(TYPES))))

    print("\n=== Per-Type Metrics ===")
    print(f"  {'Type':<12} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*33}")
    for i, t in enumerate(TYPES):
        print(f"  {t:<12} {f1_per[i]:>6.4f} {prec_per[i]:>6.4f} {rec_per[i]:>6.4f}")


def print_per_gen_metrics(y_true, y_pred, test_paths):
    from collections import defaultdict
    gen_data = defaultdict(lambda: ([], []))
    for i, path in enumerate(test_paths):
        gen = get_generation(int(Path(path).parent.name))
        gen_data[gen][0].append(y_true[i])
        gen_data[gen][1].append(y_pred[i])

    print("\n=== Per-Generation Metrics ===")
    print(f"  {'Gen':<6} {'N':>4} {'Acc':>6} {'F1':>6}")
    print(f"  {'-'*26}")
    for gen in sorted(gen_data):
        yt = np.array(gen_data[gen][0])
        yp = np.array(gen_data[gen][1])
        acc = accuracy_score(yt, yp)
        f1  = f1_score(yt, yp, average="macro", zero_division=0)
        print(f"  Gen {gen:<2}  {len(yt):>4} {acc:>6.4f} {f1:>6.4f}")


def img_to_b64(path):
    from dataset import rgba_to_rgb
    img = rgba_to_rgb(Image.open(path).convert("RGBA")).resize((96, 96))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def save_mistake_examples(y_true, y_pred, y_probs, test_paths, n=30, n_test=None):
    mistakes = []
    for i in range(len(y_true)):
        if not np.array_equal(y_true[i], y_pred[i]):
            true_types = [TYPES[j] for j in range(len(TYPES)) if y_true[i][j]]
            pred_types = [TYPES[j] for j in range(len(TYPES)) if y_pred[i][j]]
            partial = bool(np.logical_and(y_true[i], y_pred[i]).any())
            confidence = float(y_probs[i].max())
            mistakes.append((i, true_types, pred_types, confidence, partial))

    n_partial = sum(1 for *_, p in mistakes if p)
    n_wrong   = len(mistakes) - n_partial

    # full wrong first, then partial — within each group sort by confidence desc
    mistakes.sort(key=lambda x: (x[4], -x[3]))
    sample = mistakes[:n]

    cards = []
    for i, true_types, pred_types, confidence, partial in sample:
        b64 = img_to_b64(test_paths[i])
        true_str  = " / ".join(true_types)
        pred_str  = " / ".join(pred_types) if pred_types else "(none)"
        border    = "#fa0" if partial else "#e55"
        tag       = "Partial" if partial else "Wrong"
        tag_color = "#fa0" if partial else "#f66"
        cards.append(f"""
        <div style="display:inline-block;margin:8px;text-align:center;
                    border:2px solid {border};border-radius:8px;padding:6px;background:#1a1a1a">
          <img src="data:image/png;base64,{b64}" width="96" height="96"
               style="image-rendering:pixelated"/><br>
          <span style="color:#4af;font-size:12px">True: {true_str}</span><br>
          <span style="color:#f66;font-size:12px">Pred: {pred_str}</span><br>
          <span style="color:{tag_color};font-size:11px">{tag} | Conf: {confidence:.2%}</span>
        </div>""")

    total = n_test or len(y_true)
    html = f"""<!DOCTYPE html><html><body style="background:#111;color:#eee;font-family:sans-serif">
    <h2 style="padding:12px">CNN &mdash; {len(mistakes)} mistakes out of {total} test
      &nbsp;|&nbsp; <span style="color:#e55">{n_wrong} wrong</span>
      &nbsp;|&nbsp; <span style="color:#fa0">{n_partial} partial (1 of 2 types correct)</span>
      &nbsp;(showing top {len(sample)})</h2>
    <div style="padding:12px">{"".join(cards)}</div>
    </body></html>"""

    out = RESULTS_DIR / "mistakes_CNN.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")


def main():
    if not CHECKPOINT_PATH.exists():
        print(f"No checkpoint found at {CHECKPOINT_PATH}. Run train.py first.")
        sys.exit(1)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val F1: {ckpt['val_f1']:.4f})")

    model = build_model(num_classes=len(TYPES)).to(device)
    model.load_state_dict(ckpt["model_state"])

    dataset = PokemonSpriteDataset()
    _, _, test_idx = gen_gen_split(dataset.index, train_gens=(1, 2, 3), test_gens=(4, 5, 6))

    # num_workers=0 loads data in the main process — safe on Windows, lower RAM usage.
    # Increase to 2-4 on Linux/Mac or if you have spare RAM for faster data loading.
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False, num_workers=4)
    print(f"Test set size: {len(test_idx)}")

    test_paths = [dataset.index[i][0] for i in test_idx]
    n_types = [int(dataset.index[i][1].sum()) for i in test_idx]
    y_true, y_pred, y_probs = collect_predictions(model, test_loader, device, n_types)

    print_summary(y_true, y_pred, y_probs)
    print_per_gen_metrics(y_true, y_pred, test_paths)

    RESULTS_DIR.mkdir(exist_ok=True)
    np.save(RESULTS_DIR / "y_true.npy", y_true)
    np.save(RESULTS_DIR / "y_pred.npy", y_pred)
    np.save(RESULTS_DIR / "y_probs.npy", y_probs)

    print("\nGenerating mistake gallery...")
    save_mistake_examples(y_true, y_pred, y_probs, test_paths, n_test=len(test_idx))
    print(f"\nResults saved to {RESULTS_DIR}/")
    generate_report.main()


if __name__ == "__main__":
    main()
