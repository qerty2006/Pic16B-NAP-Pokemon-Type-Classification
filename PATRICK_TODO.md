# Patrick's Classification Work

## Your Role
Model selection, training, and tuning. Build and compare multiple classifiers on the sprite data, evaluate them, and run ablations.

---

## Before You Start — Check With Nish
You need the sprites to exist before any of this works. Ask Nish if he has run:
1. `Data-Acquisition/setup_pokerogue_assets.sh` — downloads sprite sheets
2. `Data-Acquisition/sprite_splitter.py` — cuts sprite sheets into individual images → `Data-Acquisition/split_sprites/`

Also need `pokeapi_data/` to exist (his script). If it doesn't, run:
```bash
python Data-Acquisition/pokeapi_data.py
```

---

## What To Build (in order)

### 1. Dataset class (`Classification/dataset.py`)
- Loads sprite images from `Data-Acquisition/split_sprites/`
- Reads type labels from `pokeapi_data/<id>_<name>/<name>.json` → `types` field
- Returns `(image_tensor, label)` pairs
- Support single-label (Type 1 only) for now — add dual-type later

### 2. Baselines (`Classification/baselines.py`)
- Flatten images → PCA → decision tree, random forest, SVM
- Quick to run, gives a comparison floor for the CNN
- Scikit-learn is fine here

### 3. CNN (`Classification/cnn_model.py`)
- Fine-tune ResNet18 (PyTorch + torchvision)
- This is your main model — everything else is compared against it

### 4. Training script (`Classification/train.py`)
- Train/val/test split (consider stratified split by type)
- Log accuracy, F1, precision, recall per epoch
- Save best checkpoint

### 5. Evaluation (`Classification/evaluate.py`)
- Accuracy, F1 (macro + per-type), precision, recall, ROC-AUC
- Confusion matrix by type — this is the most interesting output
- Ajmain handles additional visualizations from these results

### 6. Ablations (after baseline works)
- **Grayscale run** — same model, strip color → tests if shape alone is enough
- **Gen split** — train on gens 1-7, test on 8-9 → tests generalization

---

## File Layout
```
Classification/
    dataset.py       ← you build this first
    baselines.py
    cnn_model.py
    train.py
    evaluate.py
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
- Claude Code can help you write any of this — just open the project and ask
