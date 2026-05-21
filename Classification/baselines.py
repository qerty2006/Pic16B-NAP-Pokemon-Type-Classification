import sys
import base64
import io
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from dataset import _build_index, TYPES, rgba_to_rgb, gen_stratified_split
import generate_report

# 64x64 is a tunable knob — CNN uses 224x224; increase here for more detail at cost of speed
IMG_SIZE = 64
N_PCA = 50
RESULTS_DIR = Path(__file__).parent / "results"


def load_image_flat(path):
    """Load one sprite as a flat normalized float array of shape (IMG_SIZE*IMG_SIZE*3,).

    Converts RGBA to RGB, resizes to IMG_SIZE×IMG_SIZE, flattens to 1D, and scales to [0, 1].
    This is the feature vector fed into PCA and the sklearn classifiers.
    """
    img = rgba_to_rgb(Image.open(path).convert("RGBA"))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img).flatten() / 255.0


def load_all_images(index):
    """Load all sprites in parallel and return an (N, IMG_SIZE*IMG_SIZE*3) float array.

    Uses ThreadPoolExecutor to parallelize disk reads. Order matches the index list.
    """
    paths = [p for p, _ in index]
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_image_flat, paths), total=len(paths), desc="Loading sprites"))
    return np.array(results)


def get_metrics(y_true, y_pred):
    """Return a dict of accuracy, macro F1, precision, and recall for single-label predictions."""
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "F1 macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def report(name, metrics):
    """Print a model's metrics dict to stdout."""
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")


def save_comparison_chart(all_metrics):
    """Save a grouped bar chart comparing all baseline models across metrics as HTML.

    Args:
        all_metrics: dict of {model_name: {metric_name: score}} for all baseline models.
    """
    metric_names = list(next(iter(all_metrics.values())).keys())

    fig = go.Figure()
    for model_name, metrics in all_metrics.items():
        fig.add_trace(go.Bar(
            name=model_name,
            x=metric_names,
            y=[metrics[m] for m in metric_names],
            text=[f"{metrics[m]:.3f}" for m in metric_names],
            textposition="outside",
        ))

    fig.update_layout(
        title="Baseline Model Comparison",
        barmode="group",
        yaxis=dict(title="Score", range=[0, 1.05]),
        xaxis_title="Metric",
        legend_title="Model",
        height=500,
    )
    out = RESULTS_DIR / "baselines_comparison.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")


def save_confusion_matrices(all_cms):
    """Save side-by-side confusion matrix heatmaps for all baseline models as a single HTML file.

    Rows = true type, columns = predicted type. Each cell shows the raw count.
    All matrices share the same type ordering (TYPES list).
    """
    model_names = list(all_cms.keys())
    fig = make_subplots(
        rows=1, cols=len(model_names),
        subplot_titles=model_names,
    )
    for col, (_, cm) in enumerate(all_cms.items(), 1):
        fig.add_trace(
            go.Heatmap(
                z=cm,
                x=TYPES,
                y=TYPES,
                colorscale="Blues",
                showscale=(col == len(model_names)),
                hovertemplate="True: %{y}<br>Pred: %{x}<br>Count: %{z}<extra></extra>",
            ),
            row=1, col=col,
        )
    fig.update_layout(
        title="Baseline Confusion Matrices (rows=true, cols=predicted)",
        height=600,
        width=400 * len(model_names),
    )
    out = RESULTS_DIR / "baselines_confusion_matrices.html"
    fig.write_html(str(out))
    print(f"Saved: {out}")


def img_to_b64(path):
    """Load a sprite, convert to RGB, resize to 96×96, and return a base64-encoded PNG string for HTML embedding."""
    img = rgba_to_rgb(Image.open(path).convert("RGBA")).resize((96, 96))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def save_mistake_examples(model_name, y_true_int, y_pred, test_paths, y_true_multihot, n=24):
    """Generate an HTML gallery of misclassified sprites for one baseline model and save to results/.

    Baselines predict a single type (argmax), so a mistake is when the predicted type
    doesn't match the primary type. A mistake is marked "Partial" if the predicted type
    happens to be the Pokemon's secondary type (i.e. it got the wrong one of its two types).
    Samples up to n examples randomly.

    Args:
        model_name: Display name used in the HTML title and output filename.
        y_true_int: (N,) array of true primary type indices.
        y_pred: (N,) array of predicted type indices.
        test_paths: List of sprite file paths, aligned with y_true_int.
        y_true_multihot: (N, 18) multi-hot array used to detect partial matches.
    """
    n_test = len(y_true_int)
    mistakes = []
    for i in range(n_test):
        if y_true_int[i] != y_pred[i]:
            all_true = [TYPES[j] for j in range(len(TYPES)) if y_true_multihot[i][j]]
            partial = y_true_multihot[i][y_pred[i]] == 1.0  # predicted type IS one of the true types
            mistakes.append((i, all_true, TYPES[y_pred[i]], partial))

    n_partial = sum(1 for *_, p in mistakes if p)
    n_wrong   = len(mistakes) - n_partial

    rng = np.random.default_rng(42)
    sample = mistakes if len(mistakes) <= n else [mistakes[i] for i in rng.choice(len(mistakes), n, replace=False)]

    cards = []
    for i, true_types, pred_type, partial in sample:
        b64 = img_to_b64(test_paths[i])
        true_str  = " / ".join(true_types)
        border    = "#fa0" if partial else "#e55"
        tag       = "Partial" if partial else "Wrong"
        tag_color = "#fa0" if partial else "#f66"
        cards.append(f"""
        <div style="display:inline-block;margin:8px;text-align:center;
                    border:2px solid {border};border-radius:8px;padding:6px;background:#1a1a1a">
          <img src="data:image/png;base64,{b64}" width="96" height="96"
               style="image-rendering:pixelated"/><br>
          <span style="color:#4af;font-size:12px">True: {true_str}</span><br>
          <span style="color:#f66;font-size:12px">Pred: {pred_type}</span><br>
          <span style="color:{tag_color};font-size:11px">{tag}</span>
        </div>""")

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    html = f"""<!DOCTYPE html><html><body style="background:#111;color:#eee;font-family:sans-serif">
    <h2 style="padding:12px">{model_name} &mdash; {len(mistakes)} mistakes out of {n_test} test
      &nbsp;|&nbsp; <span style="color:#e55">{n_wrong} wrong</span>
      &nbsp;|&nbsp; <span style="color:#fa0">{n_partial} partial (predicted secondary type)</span>
      &nbsp;(showing {len(sample)})</h2>
    <div style="padding:12px">{"".join(cards)}</div>
    </body></html>"""

    out = RESULTS_DIR / f"mistakes_{safe_name}.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved: {out}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Building index...")
    index = _build_index()

    print(f"Loading {len(index)} sprites (threaded)...")
    X = load_all_images(index)
    # argmax gives primary type only — baselines are single-label, unlike the CNN which predicts multi-hot
    y = np.array([label.argmax() for _, label in index])
    y_multihot = np.array([label for _, label in index])

    train_idx, val_idx, test_idx = gen_stratified_split(index)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test,  y_test  = X[test_idx],  y[test_idx]
    y_test_multihot  = y_multihot[test_idx]
    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

    # n_components=50 is tunable — higher captures more detail but slows SVM training
    print(f"\nFitting PCA (n={N_PCA})...")
    pca = PCA(n_components=N_PCA, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    print(f"Variance explained: {pca.explained_variance_ratio_.sum():.2%}")

    models = [
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
        ("SVM (RBF)",     SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
    ]

    test_paths = [index[i][0] for i in test_idx]

    all_metrics, all_cms = {}, {}
    for name, model in models:
        print(f"\nTraining {name}...")
        model.fit(X_train_pca, y_train)
        preds = model.predict(X_test_pca)
        metrics = get_metrics(y_test, preds)
        report(name, metrics)
        all_metrics[name] = metrics
        all_cms[name] = confusion_matrix(y_test, preds, labels=list(range(len(TYPES))))
        save_mistake_examples(name, y_test, preds, test_paths, y_test_multihot)

    import json
    metrics_out = RESULTS_DIR / "baselines_metrics.json"
    with open(metrics_out, "w") as f:
        json.dump({k: {mk: float(mv) for mk, mv in v.items()} for k, v in all_metrics.items()}, f, indent=2)
    print(f"Saved: {metrics_out}")

    print("\nSaving visualizations...")
    save_comparison_chart(all_metrics)
    save_confusion_matrices(all_cms)

    print("\nDone.")
    generate_report.main()


if __name__ == "__main__":
    main()
