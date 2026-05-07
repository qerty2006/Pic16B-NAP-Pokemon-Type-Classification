# Patrick's Classification Work

## Your Role
Model selection, training, and tuning. Build and compare multiple classifiers on the sprite data, evaluate them, and run ablations.

---

## Before You Start — Check With Nish
You need the sprites to exist before any of this works. Ask Nish if he has run:
1. ~~`Data-Acquisition/setup_pokerogue_assets.sh` — downloads sprite sheets~~ ✅
2. ~~`Data-Acquisition/sprite_splitter.py` — cuts sprite sheets into individual images → `Data-Acquisition/split_sprites/`~~ ✅

Also need `pokeapi_data/` to exist (his script). If it doesn't, run:
```bash
python Data-Acquisition/pokeapi_data.py
```
~~`pokeapi_data/` exists~~ ✅

---

## What To Build (in order)

### ~~1. Dataset class (`Classification/dataset.py`)~~ ✅
- ~~Loads sprite images from `Data-Acquisition/split_sprites/`~~
- ~~Reads type labels from `pokeapi_data/<id>_<name>/<name>.json` → `types` field~~
- ~~Returns `(image_tensor, label)` pairs~~
- ~~Support single-label (Type 1 only) for now — add dual-type later~~
- ~~Fixed: RGBA → RGB alpha composite on white, ImageNet normalization~~
- ~~Shared: `GEN_RANGES`, `get_generation`, `gen_stratified_split` live here for all files to import~~
- Run `python Classification/dataset.py` for sample output (index size, label distribution, one batch shape)

### ~~2. Baselines (`Classification/baselines.py`)~~ ✅
- ~~Flatten images → PCA → decision tree, random forest, SVM~~
- ~~Quick to run, gives a comparison floor for the CNN~~
- ~~Scikit-learn is fine here~~
- Run `python Classification/baselines.py` — prints accuracy/F1/precision/recall for each model

### ~~3. CNN (`Classification/cnn_model.py`)~~ ✅
- ~~Fine-tune ResNet18 (PyTorch + torchvision)~~
- ~~This is your main model — everything else is compared against it~~
- ~~Alpha composite sprites onto white background before passing to ResNet~~
- Run `python Classification/cnn_model.py` for a forward-pass sanity check (random input → output shape)

### ~~4. Training script (`Classification/train.py`)~~ ✅
- ~~Stratified split by generation — 70/15/15 train/val/test from each gen~~
- ~~WeightedRandomSampler — handles type imbalance~~
- ~~Log accuracy, F1, precision, recall per epoch~~
- ~~Save best checkpoint to `Classification/checkpoints/best.pt`~~
- Run `python Classification/train.py --epochs 1` to sanity check, then `--epochs 30` for full training

### ~~5. Evaluation (`Classification/evaluate.py`)~~ ✅
- ~~Accuracy, F1 (macro + per-type), precision, recall, ROC-AUC~~
- ~~Confusion matrix by type — this is the most interesting output~~
- ~~Saves `confusion_matrix.npy`, `y_true.npy`, `y_pred.npy`, `y_probs.npy` to `Classification/results/` for Ajmain~~
- Run `python Classification/evaluate.py` — loads best checkpoint and prints full report

### 6. Ablations (after baseline works)
- **Grayscale run** — same model, strip color → tests if shape alone is enough
- **Gen split** — train on gens 1-7, test on 8-9 → tests generalization

---

## File Layout
```
Classification/
    dataset.py       ← shared module (index, split logic, transforms)
    baselines.py     ← run directly
    cnn_model.py     ← imported by train.py
    train.py         ← run directly
    evaluate.py      ← run directly
    checkpoints/     ← saved by train.py (gitignored)
```

---

## Key Numbers
- 1025 Pokemon, 18 types, up to 2 types each
- Start with Type 1 only (single-label, 18 classes) — simpler
- Add dual-type (multi-label) later if time permits

---

## Metrics To Report
- Accuracy
- Precision, Recall, F1 (macro-averaged)
- ROC-AUC
- Confusion matrix (heatmap by type)

---

## Notes
- Pixel sprites are small (~96x96 or smaller) — ResNet18 is fine, no need for anything bigger
- The greyscale ablation is important for the paper's argument about whether design is intentional
- Consider using all animation frames per Pokemon as separate training samples (~10x more data, helps rare types)
- Claude Code can help you write any of this — just open the project and ask
