# Pokemon Type Classification

PIC16B NAP project for classifying Pokemon type labels from sprite images. The
main model fine-tunes EfficientNet-B0 for 18-way multi-label prediction, so a
sprite can be assigned one or two Pokemon types. Traditional PCA + sklearn
baselines are included for comparison.

Sprite data comes from PokeRogue assets. Labels come from PokeAPI.

## Disk Space Warning

The raw data and generated outputs are intentionally not committed. Plan for
about 1 GB of free space.

- Raw sprite sheets: `Data-Acquisition/pokerogue_sprites/`
- Split sprites: `Data-Acquisition/split_sprites/`
- PokeAPI JSON: `Data-Acquisition/pokeapi_data/`
- CNN checkpoints: `Classification/checkpoints/`
- Reports and arrays: `Classification/results/`

After splitting sprites, `Data-Acquisition/pokerogue_sprites/` can be deleted if
you need to recover space.

## Quick Start

Run these commands from the project root unless noted otherwise.

```bash
bash install_hooks.sh
pip install -r requirements.txt

bash Data-Acquisition/setup_pokerogue_assets.sh
python Data-Acquisition/sprite_splitter.py
python Data-Acquisition/pokeapi_data.py

python Classification/dataset.py
python Classification/baselines.py
python Classification/train.py --epochs 30
python Classification/evaluate.py
python Classification/generate_report.py
```

The maintained all-in-one runner is also available:

```bash
python Classification/pipeline.py --mode all
```

## Environment Setup

Create a local conda environment if you do not already have one:

```bash
conda create --yes --prefix ./.conda python=3.10
conda activate ./.conda
pip install -r requirements.txt
```

CPU-only PyTorch is installed from `requirements.txt`. For CUDA, install the
right PyTorch build from <https://pytorch.org/get-started/locally/> first, then
run `pip install -r requirements.txt` for the rest.

On Windows, use Git Bash for `.sh` scripts. If `conda` is not recognized in
PowerShell, open Anaconda PowerShell Prompt or run `conda init powershell`, then
restart the terminal.

## Data Preparation

Download and split sprites:

```bash
bash Data-Acquisition/setup_pokerogue_assets.sh
python Data-Acquisition/sprite_splitter.py
```

Fetch labels and variety data:

```bash
python Data-Acquisition/pokeapi_data.py
```

Faster threaded fetch:

```bash
python Data-Acquisition/pokeapi_data_threaded.py
```

The data scripts now write under `Data-Acquisition/` no matter where they are
called from. The dataset reads:

- `Data-Acquisition/split_sprites/`
- `Data-Acquisition/pokeapi_data/`

If you change sprite or label loading logic, delete
`Classification/.index_cache.pkl` or bump `CACHE_VERSION` in
`Classification/dataset.py` before rebuilding the dataset.

## Canonical Workflow

Sanity-check dataset indexing and model shape:

```bash
python Classification/dataset.py
python Classification/cnn_model.py
```

Run baselines:

```bash
python Classification/baselines.py
```

Train the CNN:

```bash
python Classification/train.py --epochs 30
```

Evaluate the CNN and generate the report:

```bash
python Classification/evaluate.py
python Classification/generate_report.py
```

The shared CNN prediction rule lives in `Classification/prediction.py`: always
predict the highest-probability type, and also predict the second-highest type
when the probability gap is less than `0.25`.

## Outputs

All maintained outputs are written to `Classification/results/`.

| File | Description |
| --- | --- |
| `index.html` | Combined report for CNN and baseline metrics. |
| `baselines_comparison.html` | Bar chart comparing baseline metrics. |
| `baselines_confusion_matrices.html` | Baseline confusion matrix heatmaps. |
| `baselines_metrics.json` | Raw baseline metrics read by `generate_report.py`. |
| `mistakes_CNN.html` | CNN misclassification gallery. |
| `mistakes_<Model>.html` | Baseline misclassification galleries. |
| `y_true.npy`, `y_pred.npy`, `y_probs.npy` | CNN evaluation arrays. |
| `training_log.csv` | Training and validation metrics by epoch. |

Checkpoints are saved to `Classification/checkpoints/best.pt`.

## Data Visualizations

Generate type-distribution plots:

```bash
python run_analysis.py
```

Or run the visualizer directly:

```bash
python Data-Analysis/pokeapi_visualizers.py
```

## Metrics

| Metric | Meaning |
| --- | --- |
| Accuracy | Exact match between predicted and true multi-hot labels. |
| F1 macro | F1 averaged equally across all 18 types. This is the primary metric. |
| Precision | How often predicted types are correct. |
| Recall | How often true types are recovered. |
| ROC-AUC | How well probabilities separate each type from all others. |

## Repository Layout

- `Classification/`: maintained dataset, model, train, evaluate, baseline, and report scripts.
- `Data-Acquisition/`: sprite and PokeAPI data scripts.
- `Data-Analysis/`: exploratory data visualizations.
- `tests/`: lightweight tests that do not require downloaded data.
- `archive/`: older experiments and personal helper scripts kept for reference.

Generated data, checkpoints, and reports are ignored by git.
