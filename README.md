# Pic16B-NAP-Pokemon-Type-Classification

## Setup

### 0. Install Git Hooks (run once after cloning)

Installs a pre-push safety check that automatically runs before every `git push` and aborts if any large data files (sprites, pokeapi data, checkpoints, results) are accidentally tracked. If it catches something, it tells you exactly which files to untrack.

```bash
bash install_hooks.sh
```

To manually untrack a file it catches:
```bash
git rm -r --cached <file>
git commit -m "Remove accidentally tracked file"
```

### 1. Requirements

Ensure you have Python 3.10+ installed. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

This installs CPU-only PyTorch by default. If you have a compatible NVIDIA GPU and want CUDA support, install PyTorch separately first — visit [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) to configure the right command for your OS and CUDA version, as the link below may be outdated:

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Then run `pip install -r requirements.txt` for the remaining packages.

### 2. Download Sprites

Run the setup script to download Pokémon sprite assets from the PokéRogue repository (requires Git Bash on Windows):

```bash
bash Data-Acquisition/setup_pokerogue_assets.sh
```

Then split the sprite sheets into individual images:

```bash
python Data-Acquisition/sprite_splitter.py
```

> **Note:** If the script can't find `pokerogue_sprites/`, you may need to manually move or copy the `pokerogue_sprites/` folder into `Data-Acquisition/`. The splitter checks both the project root and `Data-Acquisition/` automatically, but the `.sh` script may place it at the root depending on where it was run from.

## Usage

### One-Click Analysis

To fetch all data and generate the default visualizations in one go:

```bash
python run_analysis.py
```

### Manual Execution

1. **Data Acquisition**:
   Fetch the latest Pokémon data:
   ```bash
   python Data-Acquisition/pokeapi_data.py
   ```
   > **Experimental:** A threaded version is available for faster downloads (~20x speedup). Re-run safe — skips completed entries and only fetches missing varieties. May occasionally hit PokeAPI rate limits; failed entries are printed but won't stop the run.
   > ```bash
   > python Data-Acquisition/pokeapi_data_threaded.py
   > ```

2. **Run Visualizations**:
   Generate the interactive HTML plots:
   ```bash
   python Data-Analysis/pokeapi_visualizers.py
   ```

## Model Training

Run these in order from the project root. `dataset.py` is a shared module — do not run it directly.

### 0. Sanity Checks (run first)
Verify data loading and model architecture before training:
```bash
python Classification/dataset.py
python Classification/cnn_model.py
```

### 1. Baselines (sklearn — fast, no GPU needed)
```bash
python Classification/baselines.py
```
Runs PCA + Decision Tree, Random Forest, and SVM on the sprite data. Prints accuracy, F1, precision, and recall for each model.

### 2. CNN Dry Run (verify pipeline, 1 epoch)
```bash
python Classification/train.py --epochs 1
```

### 3. Full CNN Training
```bash
python Classification/train.py --epochs 30
```
Fine-tunes ResNet18 on the sprite dataset. Saves the best checkpoint to `Classification/checkpoints/best.pt`. Logs per-epoch metrics to console.

### 4. Evaluate
```bash
python Classification/evaluate.py
```
Loads the best checkpoint and reports full metrics — accuracy, macro F1, precision, recall, ROC-AUC, and a confusion matrix by type. Saves results to `Classification/results/` for visualization.

---

## Visualizations

The analysis generates two primary reports:
- `generation_type_distribution.html`: A grid of 3D plots showing the type distribution within each generation.
- `type_distribution_all.html`: A comprehensive 3D plot of all Pokémon types across all generations.

---

## Understanding the Metrics

### Accuracy
The percentage of predictions that were correct. Simple but misleading when classes are imbalanced — a model that always predicts "Water" would score ~13% accuracy without learning anything.

### F1 Score (macro)
The harmonic mean of precision and recall, averaged equally across all 18 types regardless of how many Pokemon are in each type. This is the primary metric here because it penalises the model for ignoring rare types like Ice or Fairy. Higher is better.

### Precision
Of all the times the model predicted a type, how often it was actually right. Low precision means lots of false positives (e.g., calling non-Fire Pokemon "Fire").

### Recall
Of all the Pokemon of a given type, how many the model correctly identified. Low recall means the model is missing examples of that type.

### ROC-AUC
Measures how well the model separates each type from all others across different confidence thresholds. Ranges from 0.5 (random) to 1.0 (perfect). Reported macro-averaged across all types.

---

## Understanding the Output Files

All outputs are saved to `Classification/results/`.

### `baselines_comparison.html`
Grouped bar chart comparing Decision Tree, Random Forest, and SVM across accuracy, F1, precision, and recall. Use this as the floor — your CNN should beat all of these.

### `baselines_confusion_matrices.html`
Side-by-side heatmaps for each baseline model. Rows = true type, columns = predicted type. A perfect model has a bright diagonal. Off-diagonal cells show what the model confused — e.g., confusing Poison for Grass is a common and interpretable mistake.

### `mistakes_<Model>.html`
Gallery of misclassified sprites for each baseline model. Each card shows the actual sprite image with the true type and what the model predicted instead. Useful for seeing whether mistakes are reasonable (e.g., visually ambiguous Pokemon) or random.

### `confusion_matrix.npy` / `y_true.npy` / `y_pred.npy` / `y_probs.npy`
Raw numpy arrays from CNN evaluation, passed to Ajmain for further visualizations.
