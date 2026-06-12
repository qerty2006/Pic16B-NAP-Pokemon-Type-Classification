# Pokémon Type Classification — PIC16B NAP Project

Classifies Pokémon types from sprite images using traditional ML baselines and a fine-tuned EfficientNet-B0 CNN. Supports dual-type prediction — the model predicts both types for dual-type Pokémon. Sprite data sourced from PokéRogue; type labels from PokéAPI.

---

## Disk Space Warning

This project uses large data files that are not committed to the repo. You will need **~1GB free** to run everything comfortably. If you're running low:

- Delete `Classification/pokerogue_sprites/` after running `sprite_splitter.py` — the raw sprite sheets are no longer needed once split
- Clear the PyTorch model cache: `Remove-Item -Recurse -Force "$env:USERPROFILE\.cache\torch"` (it re-downloads automatically when needed)
- Run Windows Disk Cleanup: `cleanmgr` in PowerShell

---

## Quick Start

First set the cd of the terminal to the home directory and then run. This is important!

```bash
bash install_hooks.sh          # one-time setup — installs pre-push safety check
pip install -r requirements.txt
bash Data-Acquisition/setup_pokerogue_assets.sh
python Data-Acquisition/sprite_splitter.py
python Data-Acquisition/pokeapi_data_threaded.py
python Classification/dataset.py      # sanity check
python Classification/baselines.py
python Classification/train.py --epochs 30
python Classification/evaluate.py
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

> **Note:** Make sure you run the setup script from the project root directory so that it downloads the sprites into `Classification/pokerogue_sprites/` correctly.

> **Note:** The splitter script automatically searches both `Classification/pokerogue_sprites/` and `Data-Acquisition/pokerogue_sprites/` for the raw assets.

### 5. Fetch Pokémon Data

```bash
python Data-Acquisition/pokeapi_data_threaded.py
```

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
- `combined_generation_type_distribution.html` — Interactive 3D surface plot with dropdown selectors for each generation
- `generation_type_combo_evolution.html` — Stacked bar chart showing type combination count evolution over generations
- `top_20_type_combos_gen_0.html` — Horizontal bar chart of the top 20 most frequent type combinations

---

## Easy Run Guide

Once all the data is extracted and set-up is complete, you can run `Classification/pipeline.py` to begin running the model. 
`Classification/pipeline.py` essentially compiles all the important functions into one for easy use. While we still keep
other files so we can work on it in the future, please just run `Classification/pipeline.py` as it has the most up-to-date
and clean code.

Below are some important parameters and factors you can switch around as you like to test the model in various
ways. Note that you can either change it manually in each section or, our suggestion, run it on the command line
with the relevant flag followed by the parameter value you want. For example:

Also please remember to have your cd set to the home directory!

```bash
python Classification/pipeline.py --split "generation"
```

This essentially runs the pipeline with split set to generation. You can manually change this in the
section below.


```bash
 parser.add_argument(
        "--split", 
        type=str, 
        default="stratified",
        choices=["stratified", "generation"],
        help="Type of split strategy to use: 'stratified' (default) or 'generation'"
    )
```
By default, our pipeline code uses the "stratified" split which trains and tests on a train-test-split across
all generations. For generation wise analysis switch to "generation" and augment the train and test gens
in the following portion as you like.

```bash
    parser.add_argument(
        "--train-gens", 
        type=int, 
        nargs="+", 
        default=[1, 2, 3],
        help="Generations to train on for 'generation' split (default: 1 2 3)"
    )
    parser.add_argument(
        "--test-gens", 
        type=int, 
        nargs="+", 
        default=[4, 5, 6],
        help="Generations to test on for 'generation' split (default: 4 5 6)"
    )
```

You can alter test and validation fractions in the sections given below: 

```bash
   parser.add_argument(
        "--val-frac", 
        type=float, 
        default=0.15, 
        help="Fraction of validation samples (default: 0.15)"
    )
    parser.add_argument(
        "--test-frac", 
        type=float, 
        default=0.25, 
        help="Fraction of test samples for stratified split (default: 0.25)"
    )
```

Some other model parameters you can change can be altered in the sections below (along with the
option to use scratch CNN instead of EfficientNet-B0.

```bash

    parser.add_argument(
        "--model-type",
        type=str,
        default="efficientnet",
        choices=["efficientnet", "scratch"],
        help="Model architecture: 'efficientnet' (default) or 'scratch'"
    )

    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed for splitting data (default: 42)"
    )
    
    # Training configurations
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=30, 
        help="Number of epochs to train (default: 30)"
    )
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=32, 
        help="Batch size (default: 32)"
    )
    parser.add_argument(
        "--lr", 
        type=float, 
        default=1e-4, 
        help="Learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--freeze-backbone", 
        action="store_true", 
        help="Freeze backbone and only train the classifier head"
    )
```

Finally, incase you want to check with greyscale, you can just run the code in the command line with
--greyscale

The relevant section is given below:

```bash
 parser.add_argument(
        "--grayscale",
        action="store_true",
        help="Convert the dataset to grayscale before passing it to the model"
    )
```

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

4. Evaluation (evaluate.py / pipeline.py)
   → sigmoid(logits) → probabilities [18]
   → Inference Gap Threshold: Always predict the highest probability type. If the difference between the 1st and 2nd highest probability is less than 0.25 (GAP_THRESHOLD), also predict the 2nd highest type as well.
   → Compare predicted types vs true [grass, poison] → correct if both match exactly
```

---

## Understanding the Metrics

| Metric | What it means |
|---|---|
| **Accuracy** | **Partial Match Accuracy** ("At Least 1 Right"). The percentage of Pokémon where the model predicts at least one of the true types correctly. This is the definition of accuracy used in our final report, as identifying even one correct type is highly valuable for dual-type Pokémon. |
| **F1 (macro)** | Harmonic mean of precision and recall, calculated per-class and averaged equally across all 18 types. Our primary metric — penalizes ignoring rare types like Ice or Fairy. |
| **Precision** | Macro-averaged precision. Of all times the model predicted a type, how often it was right. Low = too many false positives. |
| **Recall** | Macro-averaged recall. Of all Pokémon of a given type, how many were correctly found. Low = model is missing that type. |
| **ROC-AUC** | Macro-averaged Area Under the ROC Curve. Evaluates how well the model's raw probability scores separate each type from all others, independent of the decision threshold. 0.5 = random, 1.0 = perfect. |

---

All outputs are saved to `Classification/results/` (and baseline outputs to `Patrick/results/` or `Classification/results/`).

## Output Files

| File / Pattern | Description |
|---|---|
| `baselines_comparison.html` | Grouped bar chart comparing all three baseline models across all metrics. |
| `baselines_confusion_matrices.html` | Side-by-side confusion matrix heatmaps for each baseline. Rows = true type, cols = predicted. |
| `baselines_metrics.json` | Raw baseline metrics (accuracy, F1, precision, recall). |
| `mistakes_<Model>.html` | Gallery of misclassified sprites for baseline models. Red border = fully wrong, orange = partial. |
| `mistakes_<color/grayscale>_gallery.html` | Detailed diagnostic HTML mistake gallery generated by `pipeline.py` (with model confidence breakdowns). |
| `mistakes_<color/grayscale>_CNN.html` | Mistake gallery generated by `evaluate.py`. |
| `training_log_<model_type>_<color/grayscale>.csv` | Epoch-by-epoch loss, accuracy, F1, precision, and recall metrics for both train and validation phases. |
| `split_config_<model_type>_<color/grayscale>.json` | Train, validation, and test indices used for the run to ensure reproducibility. |
| `y_true_<model_type>_<color/grayscale>.npy`<br>`y_pred_<model_type>_<color/grayscale>.npy`<br>`y_probs_<model_type>_<color/grayscale>.npy` | Ground truth (multi-hot), predictions (multi-hot), and sigmoid probabilities from the CNN test set. |

## Member Contributions

**Nishanth:** 
- Data acquisition: wrote all scripts related to getting Pokerogue and PokeAPI data (Patrick modified them to allow multithreading).
- Suggested the EfficientNet-B0 model; implemented the Scratch CNN; conducted grayscale analysis; implemented feature visualization.
- Cowrote analysis with Ajmain, and wrote about the results of our project that note the importance of certain scientific practices.

**Ajmain:** 
- Created the code for splitting our source data into individual sprites and the original code to implement one-hot encoding for types. 
- Implemented and tested the ViT-B/16 architecture.
- Implemented the inference gap threshold.
- Selected and reported evaluation metrics. Tested with different hyperparameters, augmentations, sample sizes, and data segments.
- Cowrote analysis with Nishanth.

**Patrick:** 
Developed the flat-feature classifiers (Decision Tree, Random Forest, SVM) and the initial EfficientNet-B0 pipeline; designed the weighted sampler and co-designed the inference gap threshold.

## OUTDATED: Model Training (USE EASY RUN GUIDE ABOVE)

**NOTE** The intructions here are outdated and simply for self-reference. Please follow Easy Run Guide above.

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
Loads the best checkpoint and prints full metrics. Uses the inference gap threshold to predict types. Saves results and a mistake gallery to `Classification/results/`.

---

