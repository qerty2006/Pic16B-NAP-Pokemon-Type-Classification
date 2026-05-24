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

import base64
from evaluate import img_to_b64
from dataset import PokemonSpriteDataset
from dataset import gen_stratified_split
RESULTS_DIR = Path(__file__).parent / "results"

dataset = PokemonSpriteDataset()
_, _, test_idx = gen_stratified_split(dataset.index)
test_paths = [dataset.index[i][0] for i in test_idx]
y_true  = np.load(RESULTS_DIR / "y_true.npy")
y_pred  = np.load(RESULTS_DIR / "y_pred.npy")
y_probs = np.load(RESULTS_DIR / "y_probs.npy")


def save_all_mistake_examples_with_probs(y_true, y_pred, y_probs, test_paths, n_test=None):
    """Generate an HTML gallery of every single CNN mistake, displaying detailed

    probabilities for every type the model was heavily considering.
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
    n_wrong = len(mistakes) - n_partial

    # Sort: completely wrong first, then partial matches
    mistakes.sort(key=lambda x: (x[4], -x[3]))
    sample = mistakes

    cards = []
    for i, true_types, pred_types, confidence, partial in sample:
        b64 = img_to_b64(test_paths[i])
        true_str = " / ".join(true_types)
        pred_str = " / ".join(pred_types) if pred_types else "(none)"
        border = "#fa0" if partial else "#e55"
        tag = "Partial" if partial else "Wrong"
        tag_color = "#fa0" if partial else "#f66"

        # --- NEW: Build the detailed type breakdown breakdown list ---
        prob_rows_html = ""

        # Sort ALL 18 types by what the model scored highest for this specific sample
        type_prob_pairs = [(TYPES[j], float(y_probs[i][j]), bool(y_pred[i][j])) for j in range(len(TYPES))]
        type_prob_pairs.sort(key=lambda x: -x[1])  # Sort descending by probability value

        for t_name, score, is_predicted in type_prob_pairs:
            # Only display types that the model gave > 1.0% confidence to
            # This keeps the layout incredibly neat and focused on relevant data
            if score >= 0.01:
                # Highlight text if the model actually picked this type via threshold rules
                highlight_style = "color:#ff4a9e; font-weight:bold;" if is_predicted else "color:#aaa;"

                prob_rows_html += f"""
                <div style="display:flex; justify-content:space-between; font-size:11px; margin:2px 0; {highlight_style}">
                  <span>{t_name}</span>
                  <span>{score:.1%}</span>
                </div>
                """
        # -------------------------------------------------------------

        cards.append(f"""
        <div style="display:inline-block; margin:8px; text-align:left; width:180px;
                    border:2px solid {border}; border-radius:8px; padding:10px; background:#1a1a1a; vertical-align:top">

          <div style="text-align:center; background:#222; border-radius:4px; padding:4px;">
            <img src="data:image/png;base64,{b64}" width="96" height="96" style="image-rendering:pixelated"/><br>
          </div>

          <div style="margin-top:8px; border-bottom:1px solid #333; padding-bottom:6px; margin-bottom:6px;">
            <span style="color:#4af; font-size:12px; font-weight:bold; display:block;">True: {true_str}</span>
            <span style="color:#f66; font-size:12px; display:block;">Pred: {pred_str}</span>
            <span style="color:{tag_color}; font-size:11px; font-weight:bold; display:block; margin-top:2px;">{tag}</span>
          </div>

          <!-- Probability readout block -->
          <div style="background:#111; padding:6px; border-radius:4px;">
            <div style="font-size:10px; color:#666; text-transform:uppercase; font-weight:bold; margin-bottom:4px; border-bottom:1px solid #222;">Model Confidences</div>
            {prob_rows_html}
          </div>

        </div>""")

    total = n_test or len(y_true)
    html = f"""<!DOCTYPE html><html>
    <head>
      <meta charset="utf-8">
      <title>CNN Detailed Mistake Analysis</title>
    </head>
    <body style="background:#0d0d0d; color:#eee; font-family:sans-serif; padding:16px;">
    <h2 style="padding:12px; border-bottom:1px solid #333; margin-bottom:16px;">
      CNN &mdash; {len(mistakes)} mistakes out of {total} test
      &nbsp;|&nbsp; <span style="color:#e55">{n_wrong} completely wrong</span>
      &nbsp;|&nbsp; <span style="color:#fa0">{n_partial} partial (1 of 2 correct)</span>
    </h2>
    <div style="display:flex; flex-wrap:wrap; justify-content:flex-start;">{"".join(cards)}</div>
    </body></html>"""

    out = RESULTS_DIR / "mistakes_CNN_all.html"
    out.write_text(html, encoding="utf-8")
    print(f"Saved complete diagnostic breakdown ({len(mistakes)} items) to: {out}")


if __name__ == "__main__":
    save_all_mistake_examples_with_probs(y_true, y_pred, y_probs, test_paths, n_test=None)