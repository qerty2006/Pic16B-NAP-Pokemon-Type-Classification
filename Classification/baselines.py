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

IMG_SIZE = 64
N_PCA = 50
RESULTS_DIR = Path(__file__).parent / "results"


def load_image_flat(path):
    img = rgba_to_rgb(Image.open(path).convert("RGBA"))
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return np.array(img).flatten() / 255.0


def load_all_images(index):
    paths = [p for p, _ in index]
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(load_image_flat, paths), total=len(paths), desc="Loading sprites"))
    return np.array(results)


def get_metrics(y_true, y_pred):
    return {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "F1 macro":  f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall":    recall_score(y_true, y_pred, average="macro", zero_division=0),
    }


def report(name, metrics):
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")


def save_comparison_chart(all_metrics):
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
    img = rgba_to_rgb(Image.open(path).convert("RGBA")).resize((96, 96))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def save_mistake_examples(model_name, y_true, y_pred, test_paths, n=24):
    mistakes = [(i, y_true[i], y_pred[i]) for i in range(len(y_true)) if y_true[i] != y_pred[i]]
    rng = np.random.default_rng(42)
    sample = mistakes if len(mistakes) <= n else [mistakes[i] for i in rng.choice(len(mistakes), n, replace=False)]

    cards = []
    for i, true_label, pred_label in sample:
        b64 = img_to_b64(test_paths[i])
        true_type = TYPES[true_label]
        pred_type = TYPES[pred_label]
        cards.append(f"""
        <div style="display:inline-block;margin:8px;text-align:center;
                    border:2px solid #e55;border-radius:8px;padding:6px;background:#1a1a1a">
          <img src="data:image/png;base64,{b64}" width="96" height="96"
               style="image-rendering:pixelated"/><br>
          <span style="color:#4af;font-size:12px">True: {true_type}</span><br>
          <span style="color:#f66;font-size:12px">Pred: {pred_type}</span>
        </div>""")

    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "")
    html = f"""<!DOCTYPE html><html><body style="background:#111;color:#eee;font-family:sans-serif">
    <h2 style="padding:12px">{model_name} — {len(mistakes)} mistakes (showing {len(sample)})</h2>
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
    y = np.array([label for _, label in index])

    train_idx, val_idx, test_idx = gen_stratified_split(index)
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    print(f"Split — train: {len(train_idx)}, val: {len(val_idx)}, test: {len(test_idx)}")

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
        save_mistake_examples(name, y_test, preds, test_paths)

    print("\nSaving visualizations...")
    save_comparison_chart(all_metrics)
    save_confusion_matrices(all_cms)

    print("\nDone.")


if __name__ == "__main__":
    main()
