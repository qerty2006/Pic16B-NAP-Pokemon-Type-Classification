# Pokémon Type Classification — PIC16B NAP Project

Classifies Pokémon types from sprite images using traditional ML baselines and a fine-tuned EfficientNet-B0 CNN. Supports dual-type prediction — the model predicts both types for dual-type Pokémon. Sprite data sourced from PokéRogue; type labels from PokéAPI.

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
python Classification/dataset.py      # sanity check
python Classification/baselines.py
python Classification/train.py --epochs 30
python Classification/evaluate.py
python Classification/generate_report.py  # results/index.html
```

---

## Setup

### 1. Create & Activate the Conda Environment

The project uses a local conda environment stored in `.conda/`.

**Create it (first time only):**
```bash
conda create --yes --prefix ./.conda python
```

**Activate it (every session):**
```bash
conda activate ./.conda
```

> **Windows note:** If `conda` is not recognised in PowerShell, open an Anaconda PowerShell prompt or run `conda init powershell` once, then restart the terminal.

Once activated your prompt will show `(.conda)`. All subsequent steps assume the environment is active.

---

### 2. Install Git Hooks (run once after cloning)

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

### 3. Install Dependencies

Ensure Python 3.10+ is installed, then:

```bash
pip install -r requirements.txt
```

This installs CPU-only PyTorch by default. For GPU support, install PyTorch separately first — visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to get the right command for your CUDA version, then run `pip install -r requirements.txt` for the rest.

### 4. Download Sprites

Requires Git Bash on Windows:

```bash
bash Data-Acquisition/setup_pokerogue_assets.sh
python Data-Acquisition/sprite_splitter.py
```

> **Note:** Make sure that the cd is set to the Data Acquisition folder (in git bash) before you run the bash prompt so that it downloads the prompts to that folder. 

> **Note:** If the splitter can't find `pokerogue_sprites/`, manually move it into `Data-Acquisition/`. The script checks both the project root and `Data-Acquisition/` automatically.

### 5. Fetch Pokémon Data

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
#SUPER IMPORTANT: Right now if you make any changes to dataset.py make sure to delete .index_cache.pkl or change Cache_version before running
#Otherwise, a new dataset comprised of your changes will not be created.
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
Fine-tunes EfficientNet-B0 on 224×224 sprites using `BCEWithLogitsLoss` for multi-label type prediction. Saves the best checkpoint by validation F1 to `Classification/checkpoints/best.pt`.

### 4. Evaluate
```bash
python Classification/evaluate.py
```
Loads the best checkpoint and prints full metrics. Uses top-k prediction — the model is told how many types each Pokémon has and predicts exactly that many. Saves results and a mistake gallery to `Classification/results/`.

### 5. Generate Report
```bash
python Classification/generate_report.py
```
Combines CNN and baseline results into a single `Classification/results/index.html` — comparison table across all models, explanation of how each model works, and links to all output galleries. Requires both `evaluate.py` and `baselines.py` to have been run first.

---

## Data Flow Example

What happens to a single Bulbasaur sprite from raw file to prediction:

```
1. Raw sprite (RGBA PNG, ~80x80 pixels)
   → rgba_to_rgb(): paste on white background → RGB
   → Resize to 224x224, ImageNet normalize
   → Tensor shape: [3, 224, 224]

2. Label (from pokeapi_data/1_bulbasaur/bulbasaur.json)
   → types: ["grass", "poison"]
   → multi-hot encode: [0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0]  (grass=1, poison=1)
   → Tensor shape: [18]

3. Training (train.py)
   → Forward: EfficientNet-B0 → raw logits [18]
   → Loss: BCEWithLogitsLoss(logits, multi-hot label) — 18 independent binary losses
   → Backprop updates weights

4. Evaluation (evaluate.py)
   → sigmoid(logits) → probabilities [18]
   → top-k: k=2 (Bulbasaur has 2 types), pick 2 highest probs
   → Compare predicted [grass, poison] vs true [grass, poison] → correct
```

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
| `index.html` | Single-page report — model comparison table, how each model works, links to all outputs. |
| `baselines_comparison.html` | Grouped bar chart comparing all three baseline models across all metrics. |
| `baselines_confusion_matrices.html` | Side-by-side confusion matrix heatmaps for each baseline. Rows = true type, cols = predicted. |
| `baselines_metrics.json` | Raw baseline metrics (accuracy, F1, precision, recall) — read by `generate_report.py`. |
| `mistakes_<Model>.html` | Gallery of misclassified sprites. Red border = fully wrong, orange = partial (1 of 2 types correct for dual-type Pokémon). CNN version sorted by confidence. |
| `y_true.npy` / `y_pred.npy` / `y_probs.npy` | Ground truth (multi-hot), predictions (multi-hot), and sigmoid probabilities from CNN test set. |

## Contributions Section:
**Patrick:** 
Developed the flat-feature classifiers (Decision Tree, Random Forest, SVM) and the initial EfficientNet-B0 pipeline; designed the weighted sampler and co-designed the inference gap threshold.

**Nishanth:** 
Data acquisition: wrote all scripts related to getting Pokerogue and PokeAPI data (Patrick modified them to allow multithreading).
Suggested the EfficientNet-B0 model; implemented the Scratch CNN; Conducted grey-scale analysis; implemented feature visualization.

**Ajmain:** 
Created the code for splitting our source data into individual sprites and the original code to implement one-hot encoding for types. 
Implemented and tested the ViT-B/16 architecture.
Implemented the inference gap threshold.
Selected and reported evaluation metrics. Tested with different hyperparameters, augmentations, sample sizes, and data segments.

