"""
Generates results/index.html - a single-page report comparing all models.
Requires:
  - results/y_true.npy, y_pred.npy, y_probs.npy  (from evaluate.py)
  - results/baselines_metrics.json                (from baselines.py)
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from dataset import TYPES

RESULTS_DIR = Path(__file__).parent / "results"

MODEL_DESCRIPTIONS = {
    "Decision Tree": {
        "how": (
            "A tree of if/else rules learned from the training data. At each node it picks "
            "the feature (PCA component) that best separates the classes, splitting until "
            "each leaf is mostly one type. Fast and interpretable, but tends to overfit and "
            "cannot capture complex patterns."
        ),
        "pipeline": "Flatten pixels -&gt; PCA (50 components) -&gt; Decision Tree",
        "color": "#4a9eff",
    },
    "Random Forest": {
        "how": (
            "An ensemble of many decision trees, each trained on a random subset of the data "
            "and features. The final prediction is a majority vote across all trees. Much more "
            "robust than a single tree (averaging reduces overfitting) but still limited by "
            "the flat pixel representation."
        ),
        "pipeline": "Flatten pixels -&gt; PCA (50 components) -&gt; 100 Decision Trees (majority vote)",
        "color": "#4aff9e",
    },
    "SVM (RBF)": {
        "how": (
            "Finds the hyperplane in feature space that maximally separates classes. The RBF "
            "(Radial Basis Function) kernel implicitly maps features into a higher-dimensional "
            "space, letting it draw non-linear boundaries. Strong for small datasets with good "
            "features, but PCA-compressed pixels are not good features for sprites."
        ),
        "pipeline": "Flatten pixels -&gt; PCA (50 components) -&gt; SVM with RBF kernel",
        "color": "#ff9e4a",
    },
    "CNN (EfficientNet-B0)": {
        "how": (
            "A deep convolutional neural network pretrained on ImageNet and fine-tuned on "
            "Pokemon sprites. Convolutional layers detect local patterns (edges, color regions, "
            "silhouettes) and compose them into increasingly abstract features. EfficientNet-B0 "
            "uses compound scaling to balance depth, width, and resolution efficiently. "
            "The final layer outputs 18 logits (one per type) and sigmoid activation allows "
            "predicting both types simultaneously for dual-type Pokemon."
        ),
        "pipeline": "224x224 RGB -&gt; EfficientNet-B0 backbone -&gt; Linear(1280, 18) -&gt; Sigmoid -&gt; gap-threshold prediction",
        "color": "#ff4a9e",
    },
}

WHY_CNN_WINS = """
<p>The baselines share a fatal flaw: <strong>they flatten the image into a 1D vector before doing anything</strong>.
Pixel (0,0) and pixel (63,63) become unrelated features, losing all spatial structure.
PCA then compresses 12,288 dimensions down to 50, keeping only the broadest color gradients.
Fine-grained shape and texture information (exactly what distinguishes a Fire-type sprite from an Electric-type) disappears.</p>

<p>The CNN avoids this entirely. Convolution slides small filters across the image, detecting local patterns (edges, color blobs, outlines)
at every position. Deeper layers compose these into higher-level concepts like "flame shape" or "blue electric arc".
EfficientNet-B0 also starts with <strong>ImageNet pretrained weights</strong>, meaning it already knows how to detect edges,
textures, and shapes before seeing a single Pokemon. The baselines start from scratch with raw pixels.</p>

<p>Finally, the CNN outputs a probability for all 18 types simultaneously. Evaluation always keeps the highest-probability
type and keeps the second-highest type when its probability is within the shared gap threshold. The baselines are hard
single-label classifiers and can never predict a secondary type.</p>
"""


def load_cnn_metrics():
    y_true  = np.load(RESULTS_DIR / "y_true.npy")
    y_pred  = np.load(RESULTS_DIR / "y_pred.npy")
    y_probs = np.load(RESULTS_DIR / "y_probs.npy")

    n_test    = len(y_true)
    mistakes  = sum(1 for i in range(n_test) if not np.array_equal(y_true[i], y_pred[i]))
    n_partial = sum(
        1 for i in range(n_test)
        if not np.array_equal(y_true[i], y_pred[i]) and np.logical_and(y_true[i], y_pred[i]).any()
    )

    try:
        auc = roc_auc_score(y_true, y_probs, average="macro")
    except ValueError:
        auc = float("nan")

    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "F1 macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC-AUC":   auc,
        "_mistakes": mistakes,
        "_partial":  n_partial,
        "_n_test":   n_test,
    }


def metric_row(name, metrics, highlight=False):
    bg   = "background:#1e1e2e;" if highlight else ""
    bold = "font-weight:bold;"   if highlight else ""
    acc  = metrics.get("Accuracy",  float("nan"))
    f1   = metrics.get("F1 macro",  float("nan"))
    prec = metrics.get("Precision", float("nan"))
    rec  = metrics.get("Recall",    float("nan"))
    auc  = metrics.get("ROC-AUC",   float("nan"))
    mis  = metrics.get("_mistakes", None)
    par  = metrics.get("_partial",  None)
    nt   = metrics.get("_n_test",   None)

    mistake_str = f"{mis} / {nt}" if mis is not None else "N/A"
    partial_str = str(par)        if par is not None else "N/A"

    def fmt(v):
        return f"{v:.4f}" if isinstance(v, float) and not np.isnan(v) else "N/A"

    return f"""
    <tr style="{bg}{bold}">
      <td style="padding:10px 16px">{name}{"&nbsp;&#11088;" if highlight else ""}</td>
      <td style="text-align:center;padding:10px">{fmt(acc)}</td>
      <td style="text-align:center;padding:10px">{fmt(f1)}</td>
      <td style="text-align:center;padding:10px">{fmt(prec)}</td>
      <td style="text-align:center;padding:10px">{fmt(rec)}</td>
      <td style="text-align:center;padding:10px">{fmt(auc)}</td>
      <td style="text-align:center;padding:10px">{mistake_str}</td>
      <td style="text-align:center;padding:10px;color:#fa0">{partial_str}</td>
    </tr>"""


def model_card(name, info):
    return f"""
    <div style="border:1px solid {info['color']};border-radius:10px;padding:20px;margin:16px 0;background:#111">
      <h3 style="color:{info['color']};margin:0 0 8px">{name}</h3>
      <p style="color:#aaa;font-size:13px;margin:0 0 12px"><strong style="color:#888">Pipeline:</strong> {info['pipeline']}</p>
      <p style="color:#ddd;line-height:1.6;margin:0">{info['how']}</p>
    </div>"""


def main():
    cnn_files = ["y_true.npy", "y_pred.npy", "y_probs.npy"]
    has_cnn       = all((RESULTS_DIR / f).exists() for f in cnn_files)
    has_baselines = (RESULTS_DIR / "baselines_metrics.json").exists()

    if not has_cnn or not has_baselines:
        missing = [f for f in cnn_files if not (RESULTS_DIR / f).exists()]
        if not has_baselines:
            missing.append("baselines_metrics.json")
        print(f"generate_report: skipping (missing {missing})")
        return

    cnn_metrics = load_cnn_metrics()
    with open(RESULTS_DIR / "baselines_metrics.json") as f:
        baseline_metrics = json.load(f)

    all_models = {**baseline_metrics, "CNN (EfficientNet-B0)": cnn_metrics}

    header_cells = "".join(
        f'<th style="padding:10px 16px;text-align:center;background:#222">{h}</th>'
        for h in ["Model", "Accuracy", "F1 macro", "Precision", "Recall", "ROC-AUC", "Mistakes", "Partial"]
    )
    rows  = "".join(metric_row(name, m, highlight=(name == "CNN (EfficientNet-B0)")) for name, m in all_models.items())
    cards = "".join(model_card(name, MODEL_DESCRIPTIONS[name]) for name in MODEL_DESCRIPTIONS)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pokemon Type Classification - Model Report</title>
  <style>
    body {{ background:#0d0d0d; color:#eee; font-family:sans-serif; max-width:1100px; margin:0 auto; padding:32px 16px; }}
    h1 {{ color:#fff; }}
    h2 {{ color:#ccc; border-bottom:1px solid #333; padding-bottom:8px; margin-top:40px; }}
    table {{ border-collapse:collapse; width:100%; margin:16px 0; }}
    th {{ color:#aaa; font-weight:normal; }}
    tr:hover {{ background:#181828; }}
    a {{ color:#4a9eff; }}
  </style>
</head>
<body>
  <h1>Pokemon Type Classification - Model Report</h1>
  <p style="color:#888">EfficientNet-B0 (multi-label, dual-type aware) vs. flat-feature baselines on {cnn_metrics['_n_test']} test samples.</p>

  <h2>Model Comparison</h2>
  <table>
    <thead><tr>{header_cells}</tr></thead>
    <tbody>{rows}</tbody>
  </table>
  <p style="color:#666;font-size:12px">Partial = dual-type Pokemon where the model predicted 1 of 2 correct types. Baselines are single-label only.</p>

  <h2>Why Does the CNN Win?</h2>
  <div style="background:#111;border-left:3px solid #ff4a9e;padding:16px 20px;border-radius:4px;line-height:1.7">
    {WHY_CNN_WINS}
  </div>

  <h2>How Each Model Works</h2>
  {cards}

  <h2>Output Files</h2>
  <ul style="line-height:2">
    <li><a href="baselines_comparison.html">Baseline comparison chart</a></li>
    <li><a href="baselines_confusion_matrices.html">Baseline confusion matrices</a></li>
    <li><a href="mistakes_CNN.html">CNN mistake gallery</a></li>
    <li><a href="mistakes_Decision_Tree.html">Decision Tree mistake gallery</a></li>
    <li><a href="mistakes_Random_Forest.html">Random Forest mistake gallery</a></li>
    <li><a href="mistakes_SVM_RBF.html">SVM mistake gallery</a></li>
  </ul>
</body>
</html>"""

    out = RESULTS_DIR / "index.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
