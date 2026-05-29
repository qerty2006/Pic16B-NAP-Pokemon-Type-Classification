# Pokemon Type Classification Run Guide

This file is the practical README for running the project end to end. Use it
when you want the exact commands and the reason for each step.

## What This Project Does

The project predicts Pokemon type labels from sprite images.

- Labels come from PokeAPI.
- Sprite sheets come from PokeRogue assets.
- Baselines use flattened sprite pixels, PCA, and sklearn classifiers.
- The main model fine-tunes EfficientNet-B0 for multi-label prediction.
- CNN predictions use the shared rule in `Classification/prediction.py`:
  - Always predict the highest-probability type.
  - Also predict the second-highest type if its probability is within `0.25`
    of the top type.

## Which Workflow Should I Use?

Use the standalone scripts for the standard final-report workflow:

```powershell
python Classification/baselines.py
python Classification/train.py --epochs 30
python Classification/evaluate.py
python Classification/generate_report.py
```

Use `pipeline.py` for the main experiment design: generation-defined
train/test splits. It can also do custom output directories, grayscale
experiments, and quick test runs:

```powershell
python Classification/pipeline.py --mode all
```

Important difference:

- `train.py` and `evaluate.py` use the default stratified split.
- `pipeline.py` is the place to define train/test generations.
- The most important pipeline flags are `--split generation`, `--train-gens`,
  and `--test-gens`.

## One-Time Setup

These commands were already run on this machine, but they are here for a fresh
clone or a rebuilt environment.

```powershell
# Accept Anaconda channel ToS if conda asks for it.
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/msys2

# Allow PowerShell to load the conda hook.
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Create and activate the local environment.
conda create --yes --prefix ./.conda python=3.11
conda activate ./.conda

# Install dependencies and the git safety hook.
pip install -r requirements.txt
bash install_hooks.sh
```

GPU machines: install the CUDA PyTorch build before `pip install -r
requirements.txt`.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

## Every Session

Start from the project root:

```powershell
conda activate ./.conda
```

Your prompt should show `(.conda)`.

## Data Acquisition

Run once, in order, from the project root.

```powershell
bash Data-Acquisition/setup_pokerogue_assets.sh
python Data-Acquisition/sprite_splitter.py
python Data-Acquisition/pokeapi_data.py
```

Faster PokeAPI download:

```powershell
python Data-Acquisition/pokeapi_data_threaded.py
```

Expected local data folders:

- `Data-Acquisition/pokerogue_sprites/`
- `Data-Acquisition/split_sprites/`
- `Data-Acquisition/pokeapi_data/`

These folders are intentionally ignored by git.

## Sanity Checks

Check that dataset indexing works:

```powershell
python Classification/dataset.py
```

Check that the model builds and produces 18 logits:

```powershell
python Classification/cnn_model.py
```

If the dataset behavior changes, delete the cache before rerunning:

```powershell
Remove-Item Classification/.index_cache.pkl
```

## Standard Final-Report Workflow

Run baselines first. They generate comparison files used by the report.

```powershell
python Classification/baselines.py
```

Train the CNN:

```powershell
python Classification/train.py --epochs 30
```

Useful quick check before a full run:

```powershell
python Classification/train.py --epochs 1
```

Evaluate the best checkpoint:

```powershell
python Classification/evaluate.py
```

Generate the combined HTML report:

```powershell
python Classification/generate_report.py
```

Primary output:

```text
Classification/results/index.html
```

## Pipeline Workflow

The pipeline is the main way to run generation-split experiments.

Think of the pipeline command as two separate choices:

```text
--split decides which Pokemon go into train/validation/test.
--mode decides which steps to run after the split is created.
```

For this project, the key split is usually:

```powershell
--split generation --train-gens 1 2 3 --test-gens 4 5 6
```

That means:

- Train on Pokemon from generations 1, 2, and 3.
- Hold out validation data from those training generations.
- Test on Pokemon from generations 4, 5, and 6.

Every pipeline mode starts by loading the dataset and creating a split. After
that, the mode decides how far to go.

Run everything with the default stratified split:

```powershell
python Classification/pipeline.py --mode all
```

Run everything with generation-defined train/test splits:

```powershell
python Classification/pipeline.py --mode all `
  --split generation `
  --train-gens 1 2 3 `
  --test-gens 4 5 6
```

### What each mode does

`--mode prepare`

- Builds the dataset index.
- Creates train/validation/test indices.
- Saves split metadata to `Classification/results/split_config.json`.
- Stops before training.

```powershell
python Classification/pipeline.py --mode prepare
```

`--mode train`

- Builds the dataset index.
- Creates train/validation/test indices.
- Trains the CNN.
- Saves the best checkpoint.
- Stops before evaluation.

```powershell
python Classification/pipeline.py --mode train
```

`--mode evaluate`

- Builds the dataset index.
- Creates train/validation/test indices.
- Loads an existing checkpoint.
- Evaluates on the test split.
- Saves `y_true.npy`, `y_pred.npy`, and `y_probs.npy`.
- Stops before visualization.

```powershell
python Classification/pipeline.py --mode evaluate
```

`--mode visualize`

- Builds the dataset index.
- Creates train/validation/test indices.
- Loads existing evaluation arrays from the results directory.
- Creates the pipeline mistake gallery.
- Does not train or evaluate.

```powershell
python Classification/pipeline.py --mode visualize
```

`--mode all`

- Runs prepare.
- Runs train.
- Runs evaluate.
- Runs visualize.

```powershell
python Classification/pipeline.py --mode all
```

### The important split rule

If you run pipeline stages separately, pass the same split arguments every time.
The split is recreated at the start of each pipeline run.

Good:

```powershell
python Classification/pipeline.py --mode train `
  --split generation `
  --train-gens 1 2 3 `
  --test-gens 4 5 6

python Classification/pipeline.py --mode evaluate `
  --split generation `
  --train-gens 1 2 3 `
  --test-gens 4 5 6
```

Risky:

```powershell
python Classification/pipeline.py --mode train --split generation --train-gens 1 2 3 --test-gens 4 5 6
python Classification/pipeline.py --mode evaluate
```

The second command silently goes back to the default stratified split.

### Split flag basics

`--split` must be followed by one of two values:

```powershell
--split stratified
--split generation
```

This works:

```powershell
python Classification/pipeline.py --mode prepare --split generation
```

This does not work:

```powershell
python Classification/pipeline.py --mode prepare --split
```

The second command is incomplete because `--split` has no value.

### Common pipeline commands

Prepare only, using the default stratified split:

```powershell
python Classification/pipeline.py --mode prepare
```

Prepare only, using generation split:

```powershell
python Classification/pipeline.py --mode prepare `
  --split generation `
  --train-gens 1 2 3 `
  --test-gens 4 5 6
```

Train only, using generation split:

```powershell
python Classification/pipeline.py --mode train `
  --split generation `
  --train-gens 1 2 3 `
  --test-gens 4 5 6
```

Evaluate only, using the same generation split:

```powershell
python Classification/pipeline.py --mode evaluate `
  --split generation `
  --train-gens 1 2 3 `
  --test-gens 4 5 6
```

Visualize only, using the same generation split:

```powershell
python Classification/pipeline.py --mode visualize `
  --split generation `
  --train-gens 1 2 3 `
  --test-gens 4 5 6
```

Quick two-epoch test run:

```powershell
python Classification/pipeline.py --mode all --test-run
```

Custom output locations:

```powershell
python Classification/pipeline.py --mode all `
  --results-dir Classification/results_experiment `
  --checkpoint-dir Classification/checkpoints_experiment
```

Grayscale experiment:

```powershell
python Classification/pipeline.py --mode all --grayscale
```

## Generation Split Workflow

Use generation split when you want to train on some Pokemon generations and test
on others. This is useful for distribution-shift experiments.

Generation numbers:

- Gen 1: Pokemon IDs 1-151
- Gen 2: 152-251
- Gen 3: 252-386
- Gen 4: 387-493
- Gen 5: 494-649
- Gen 6: 650-721
- Gen 7: 722-809
- Gen 8: 810-905
- Gen 9: 906-1025

Some forms override their base ID generation. For example, Alola forms count as
generation 7, while Galar and Hisui forms count as generation 8.

Default generation split in the pipeline:

```powershell
python Classification/pipeline.py --mode all --split generation
```

That defaults to:

- Train: generations `1 2 3`
- Test: generations `4 5 6`
- Validation: held out from the training generations

Custom generation split:

```powershell
python Classification/pipeline.py --mode all `
  --split generation `
  --train-gens 1 2 3 4 5 `
  --test-gens 6 7 8 9
```

Another example:

```powershell
python Classification/pipeline.py --mode all `
  --split generation `
  --train-gens 1 2 3 4 `
  --test-gens 5
```

Stratified split with custom fractions:

```powershell
python Classification/pipeline.py --mode all `
  --split stratified `
  --val-frac 0.15 `
  --test-frac 0.15
```

Notes:

- Generation split is controlled by `gen_gen_split()` in
  `Classification/dataset.py`.
- Stratified split is controlled by `gen_stratified_split()` in
  `Classification/dataset.py`.
- Standalone `train.py` and `evaluate.py` currently do not accept
  `--train-gens` or `--test-gens`; use `pipeline.py` for that.
- `--train-gens` and `--test-gens` only matter when `--split generation` is set.
- `--val-frac` controls how much of the training-generation pool becomes
  validation data.
- `--test-frac` only matters for `--split stratified`.

## Output Files

Main output folder:

```text
Classification/results/
```

Important files:

- `index.html`: final combined report.
- `training_log.csv`: train/validation metrics by epoch.
- `baselines_metrics.json`: baseline metrics used by the report.
- `baselines_comparison.html`: baseline metric chart.
- `baselines_confusion_matrices.html`: baseline confusion matrices.
- `mistakes_CNN.html`: CNN mistake gallery from `evaluate.py`.
- `mistakes_<Model>.html`: baseline mistake galleries.
- `y_true.npy`, `y_pred.npy`, `y_probs.npy`: CNN evaluation arrays.

Checkpoint:

```text
Classification/checkpoints/best.pt
```

All checkpoints, results, data folders, and test-run output folders should stay
untracked.

## Data Visualizations

There are two different meanings of "visualization" in this repo.

### Model visualizations

These come from CNN predictions. They need a trained/evaluated model and are
handled by `Classification/pipeline.py --mode visualize`.

```powershell
python Classification/pipeline.py --mode visualize `
  --split generation `
  --train-gens 1 2 3 `
  --test-gens 4 5 6
```

This reads evaluation arrays from `Classification/results/` and writes a mistake
gallery. Use the same split flags that you used for training/evaluation.

### Data visualizations

These come from PokeAPI label data only. They do not use the CNN checkpoint, do
not use train/test splits, and do not need model evaluation arrays.

Generate type-distribution plots:

```powershell
python run_analysis.py
```

Direct visualizer entrypoint:

```powershell
python Data-Analysis/pokeapi_visualizers.py
```

Outputs:

- `generation_type_distribution.html`
- `type_distribution_all.html`

These scripts read `Data-Acquisition/pokeapi_data/`.

## Verification Commands

Compile all maintained Python files:

```powershell
python -m compileall Classification Data-Acquisition Data-Analysis run_analysis.py
```

Run lightweight tests:

```powershell
python -m unittest discover -s tests
```

If tests skip because dependencies are missing, activate the conda environment
and install requirements:

```powershell
conda activate ./.conda
pip install -r requirements.txt
python -m unittest discover -s tests
```

## Troubleshooting

If dataset changes do not show up:

```powershell
Remove-Item Classification/.index_cache.pkl
python Classification/dataset.py
```

If `evaluate.py` cannot find a checkpoint:

```powershell
python Classification/train.py --epochs 1
python Classification/evaluate.py
```

If report generation skips because files are missing:

```powershell
python Classification/baselines.py
python Classification/evaluate.py
python Classification/generate_report.py
```

If PowerShell cannot run conda activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
conda init powershell
```

Then restart PowerShell.

## Refactor Change Log

Date: 2026-05-29

### Canonical workflow

- Made `Classification/` the maintained workflow location.
- Promoted `Classification/baselines.py` and `Classification/generate_report.py`
  from the old `Classification/Patrick/` area.
- Kept script-based public commands for baselines, training, evaluation, report
  generation, and the pipeline.

### Shared prediction logic

- Added `Classification/prediction.py`.
- Centralized the CNN prediction rule:
  - Always predict the highest-probability type.
  - Also predict the second-highest type when the probability gap is less than
    `0.25`.
- Updated `train.py`, `evaluate.py`, and `pipeline.py` to use that shared rule.

### Script cleanup

- Cleaned stale imports, commented experiment blocks, and misleading top-k
  wording from the maintained training/evaluation path.
- Moved training logs to `Classification/results/training_log.csv`.
- Fixed invalid escape-sequence warnings in `Classification/visualize_cnn.py`.

### Data path fixes

- Updated `Data-Acquisition/setup_pokerogue_assets.sh` so it runs relative to
  its own folder.
- Updated `Data-Acquisition/sprite_splitter.py` so split sprites are always
  written to `Data-Acquisition/split_sprites/`.
- Updated `Data-Acquisition/pokeapi_data.py` and
  `pokeapi_data_threaded.py` so PokeAPI JSON is written to
  `Data-Acquisition/pokeapi_data/`.

### Archive and ignored outputs

- Added `archive/README.md`.
- Archived old experiments and personal helpers:
  - `archive/patrick_experiments/`
  - `archive/small_classif/`
  - `archive/visualizer_experiment.py`
  - `archive/visualize_cnn_copy.py`
- Removed tracked generated test-run artifacts from source control while leaving
  local ignored copies alone:
  - `Classification/checkpoints_test_run/`
  - `Classification/results_test_run/`
- Expanded `.gitignore` so future checkpoint/result directories under
  `Classification/` stay untracked.

### Tests and verification

- Added `tests/test_core_logic.py` for dataset helpers, split grouping, and
  gap-threshold prediction behavior.
- Verification run:
  - `python -m compileall Classification Data-Acquisition Data-Analysis run_analysis.py` passed.
  - `git diff --check` passed.
  - `python -m unittest discover -s tests` exited OK, but skipped because the
    active `python` environment did not have `numpy` installed.

### Untouched user-owned change

- `.vscode/settings.json` already had a local modification before the refactor
  and was intentionally left alone.
