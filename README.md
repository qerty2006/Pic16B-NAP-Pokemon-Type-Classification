# Pokémon Type Classification — PIC16B NAP Project

Classifies Pokémon types from sprite images using traditional ML baselines and a fine-tuned ResNet18 CNN. Sprite data sourced from PokéRogue; type labels from PokéAPI.

---

## Disk Space Warning

This project uses large data files that are not committed to the repo. You will need **~1GB free** to run everything comfortably. If you're running low:

- Delete `Data-Acquisition/pokerogue_sprites/` after running `sprite_splitter.py` — the raw sprite sheets are no longer needed once split
- Clear the PyTorch model cache: `Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\torch"` (it re-downloads automatically when needed)
- Run Windows Disk Cleanup: `cleanmgr` in PowerShell

---

## Quick Start

```bash
bash install_hooks.sh          # one-time setup — installs pre-push safety check
pip install -r requirements.txt
bash Data-Acquisition/setup_pokerogue_assets.sh
python Data-Acquisition/sprite_splitter.py
python Data-Acquisition/pokeapi_data.py
python Classification/dataset.py   # sanity check
python Classification/baselines.py
python Classification/train.py --epochs 30
python Classification/evaluate.py
```

---

## Setup

### 1. Install Git Hooks (run once after cloning)

Installs a pre-push safety check that runs automatically before every `git push`. It aborts the push if any large data files (sprites, pokeapi data, checkpoints, results) are accidentally tracked and tells you exactly which files to fix.

> **Note for Windows users:** If pushing from VS Code fails with a WSL error, push from PowerShell instead: `git push`

```bash
bash install_hooks.sh
```

To manually untrack a file the hook catches:
```bash
git rm -r --cached <file>
git commit -m "Remove accidentally tracked file"
```

### 2. Install Dependencies

Ensure Python 3.10+ is installed, then:

```bash
pip install -r requirements.txt
```

This installs CPU-only PyTorch by default. For GPU support, install PyTorch separately first — visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to get the right command for your CUDA version, then run `pip install -r requirements.txt` for the rest.

### 3. Download Sprites

Requires Git Bash on Windows:

```bash
bash Data-Acquisition/setup_pokerogue_assets.sh
python Data-Acquisition/sprite_splitter.py
```

> **Note:** If the splitter can't find `pokerogue_sprites/`, manually move it into `Data-Acquisition/`. The script checks both the project root and `Data-Acquisition/` automatically.

### 4. Fetch Pokémon Data

```bash
python Data-Acquisition/pokeapi_data.py
```

> **Faster option (experimental):** `python Data-Acquisition/pokeapi_data_threaded.py` — ~20x speedup using threading. Re-run safe. May occasionally hit PokeAPI rate limits; failed entries are printed but won't stop the run.

---

## Data Visualizations

Generate interactive type-distribution plots:

```bash
python run_analysis.py
```

Or manually:
```bash
python Data-Analysis/pokeapi_visualizers.py
```

Outputs:
- `generation_type_distribution.html` — 3D grid of type distributions per generation
- `type_distribution_all.html` — 3D plot across all generations

---

## Model Training

Run from the project root in order. `dataset.py` is a shared module — do not run it directly.

### 0. Sanity Checks
```bash
python Classification/dataset.py    # prints index size, type distribution, split counts, batch shape
python Classification/cnn_model.py  # prints input/output shapes and param count
```

### 1. Baselines
```bash
python Classification/baselines.py
```
Runs PCA → Decision Tree, Random Forest, and SVM. Saves comparison charts and mistake galleries to `Classification/results/`. These are the floor — the CNN should beat all of them.

### 2. CNN Dry Run
```bash
python Classification/train.py --epochs 1
```
Verifies the full training pipeline before committing to a long run.

### 3. Full CNN Training
```bash
python Classification/train.py --epochs 30
```
Fine-tunes ResNet18 (~20 min on CPU). Saves the best checkpoint by validation F1 to `Classification/checkpoints/best.pt`.

### 4. Evaluate
```bash
python Classification/evaluate.py
```
Loads the best checkpoint and prints full metrics. Saves results and a mistake gallery to `Classification/results/`.

---

## Understanding the Metrics

| Metric | What it means |
|---|---|
| **Accuracy** | % of predictions correct. Misleading with imbalanced classes — a model always predicting "Water" scores ~13% for free. |
| **F1 (macro)** | Harmonic mean of precision and recall, averaged equally across all 18 types. Primary metric here — penalises ignoring rare types like Ice or Fairy. |
| **Precision** | Of all times the model predicted a type, how often it was right. Low = too many false positives. |
| **Recall** | Of all Pokémon of a given type, how many were correctly found. Low = model is missing that type. |
| **ROC-AUC** | How well the model separates each type from all others. 0.5 = random, 1.0 = perfect. |

---

## Output Files

All outputs saved to `Classification/results/`.

| File | Description |
|---|---|
| `baselines_comparison.html` | Grouped bar chart comparing all three baseline models across all metrics. |
| `baselines_confusion_matrices.html` | Side-by-side confusion matrix heatmaps for each baseline. Rows = true type, cols = predicted. |
| `mistakes_<Model>.html` | Gallery of misclassified sprites — true type vs. predicted type for each error. CNN version sorted by model confidence. |
| `confusion_matrix.npy` | Raw confusion matrix from CNN evaluation (passed to Ajmain for visualizations). |
| `y_true.npy` / `y_pred.npy` / `y_probs.npy` | Ground truth, predictions, and softmax probabilities from CNN test set. |
