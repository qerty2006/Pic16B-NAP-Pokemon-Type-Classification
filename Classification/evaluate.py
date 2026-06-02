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
from dataset import PokemonSpriteDataset, TYPES, gen_stratified_split, get_generation, parse_folder_id
from ViT import build_vit_b16
from dataset import  gen_gen_split
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
def collect_predictions2(model, loader, device, threshold=0.5):
    """Predict types purely based on a confidence threshold (no top-k cheating)."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="Evaluating"):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()

            # --- CHANGED HERE ---
            # Instead of a loop sorting top-k, anyone who clears the threshold gets a 1.
            preds = np.zeros_like(probs, dtype=int)

            GAP_THRESHOLD = 0.25  # Set this to match your training settings

            for i in range(len(probs)):
                # Find the indices that would sort the array from highest to lowest probability
                sorted_idx = np.argsort(probs[i])[::-1]

                # 1. Always lock in the absolute #1 highest guess (guarantees no "none" predictions)
                preds[i, sorted_idx[0]] = 1

                # 2. Check if the 2nd highest guess is within the gap threshold
                if (probs[i, sorted_idx[0]] - probs[i, sorted_idx[1]]) < GAP_THRESHOLD:
                    preds[i, sorted_idx[1]] = 1
            # --------------------

            all_labels.append(labels.int().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    return (
        np.vstack(all_labels),
        np.vstack(all_preds),
        np.vstack(all_probs),
    )
def print_summary(y_true, y_pred, y_probs):
    """Print overall and per-type evaluation metrics to stdout.

    Overall metrics: exact match accuracy, macro F1, precision, recall, ROC-AUC.
    Per-type metrics: F1, precision, recall for each of the 18 types individually.
    ROC-AUC is computed on raw probabilities; all other metrics use top-k predictions.
    """
    acc  = accuracy_score(y_true, y_pred)   # exact match across both types
    f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_probs, average="macro")
    except ValueError:
        auc = float("nan")

    # Custom overlap calculations
    correct_overlap = np.logical_and(y_true, y_pred).sum(axis=1)
    true_counts = y_true.sum(axis=1)
    
    at_least_one = np.mean(correct_overlap >= 1)
    two_right_all = np.mean(correct_overlap >= 2)
    
    # Calculate for dual-type Pokémon specifically
    dual_mask = (true_counts == 2)
    two_right_dual = np.mean(correct_overlap[dual_mask] == 2) if dual_mask.any() else 0.0

    print("\n=== Overall Metrics ===")
    print(f"  Exact Match Acc:  {acc:.4f}")
    print(f"  F1 (macro):       {f1:.4f}")
    print(f"  Precision:        {prec:.4f}")
    print(f"  Recall:           {rec:.4f}")
    print(f"  ROC-AUC:          {auc:.4f}")
    print(f"  At Least 1 Right: {at_least_one:.2%}")
    print(f"  2 Types Right (All):  {two_right_all:.2%}")
    print(f"  2 Types Right (Dual): {two_right_dual:.2%} (Out of dual-type Pokemon)")

    # Naive baseline comparison (Guessing the most common type)
    most_common_idx = y_true.sum(axis=0).argmax()
    most_common_type = TYPES[most_common_idx]
    y_naive = np.zeros_like(y_true)
    y_naive[:, most_common_idx] = 1
    
    naive_acc = accuracy_score(y_true, y_naive)
    naive_overlap = np.logical_and(y_true, y_naive).sum(axis=1)
    naive_at_least_one = np.mean(naive_overlap >= 1)
    naive_two_right = np.mean(naive_overlap >= 2)
    
    print(f"\n=== Naive Baseline (Guessing '{most_common_type.capitalize()}') ===")
    print(f"  Exact Match Acc:  {naive_acc:.4f}")
    print(f"  At Least 1 Right: {naive_at_least_one:.2%}")
    print(f"  2 Types Right:    {naive_two_right:.2%}")

    f1_per   = f1_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(TYPES))))
    prec_per = precision_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(TYPES))))
    rec_per  = recall_score(y_true, y_pred, average=None, zero_division=0, labels=list(range(len(TYPES))))

    print("\n=== Per-Type Metrics ===")
    print(f"  {'Type':<12} {'F1':>6} {'Prec':>6} {'Rec':>6}")
    print(f"  {'-'*33}")
    for i, t in enumerate(TYPES):
        print(f"  {t:<12} {f1_per[i]:>6.4f} {prec_per[i]:>6.4f} {rec_per[i]:>6.4f}")


# Checks whether accuracy degrades on newer generations — a proxy for distribution shift
def print_per_gen_metrics(y_true, y_pred, test_paths):
    """Print accuracy and macro F1 broken down by generation.

    Groups test samples by their generation (derived from the sprite folder name / national dex ID)
    and computes metrics per group. A drop in later generations suggests the model learned
    generation-specific visual patterns rather than type-indicative features.
    """
    from collections import defaultdict
    gen_data = defaultdict(lambda: ([], []))
    for i, path in enumerate(test_paths):
        pokemon_id = parse_folder_id(Path(path).parent.name)
        gen = get_generation(pokemon_id) if pokemon_id else 0
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
    """Load a sprite, convert to RGB, resize to 96×96, and return a base64-encoded PNG string.

    Used to embed sprite thumbnails directly into the HTML mistake gallery
    without needing separate image files.
    """
    from dataset import rgba_to_rgb
    img = rgba_to_rgb(Image.open(path).convert("RGBA")).resize((96, 96))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def save_mistake_examples(y_true, y_pred, y_probs, test_paths, n=30, n_test=None):
    """Generate an HTML gallery of misclassified CNN test sprites and save to results/.

    Sorts mistakes so fully-wrong predictions appear first, then partial (1 of 2 types correct),
    both groups ordered by descending confidence. Shows the top n examples.
    Red border = fully wrong, orange = partial match.
    """
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

    #model = build_vit_b16(num_classes=len(TYPES), freeze_backbone=True).to(device)
    model = build_model(num_classes=len(TYPES)).to(device)
    model.load_state_dict(ckpt["model_state"])

    dataset = PokemonSpriteDataset()
    _, _, test_idx = gen_stratified_split(dataset.index)
    #We use gen_stratified as base and gen_gen for extra analysis
    #_, _, test_idx = gen_gen_split(dataset.index, train_gens=(1, 2, 3), test_gens=(4, 5, 6))

    # num_workers=0 loads data in the main process — safe on Windows, lower RAM usage.
    # Increase to 2-4 on Linux/Mac or if you have spare RAM for faster data loading.
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=32, shuffle=False, num_workers=0)
    print(f"Test set size: {len(test_idx)}")

    test_paths = [dataset.index[i][0] for i in test_idx]
    n_types = [int(dataset.index[i][1].sum()) for i in test_idx]
    y_true, y_pred, y_probs = collect_predictions2(model, test_loader, device)

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
