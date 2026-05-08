# Patrick's Classification Work

## Your Role
Model selection, training, and tuning. Build and compare multiple classifiers on the sprite data, evaluate them, and run ablations.

---

## Before You Start â€” Check With Nish
You need the sprites to exist before any of this works. Ask Nish if he has run:
1. ~~`Data-Acquisition/setup_pokerogue_assets.sh` â€” downloads sprite sheets~~ âœ…
2. ~~`Data-Acquisition/sprite_splitter.py` â€” cuts sprite sheets into individual images â†’ `Data-Acquisition/split_sprites/`~~ âœ…

Also need `pokeapi_data/` to exist (his script). If it doesn't, run:
```bash
python Data-Acquisition/pokeapi_data.py
```
~~`pokeapi_data/` exists~~ âœ…

---

## What To Build (in order)

### ~~1. Dataset class (`Classification/dataset.py`)~~ âœ…
- ~~Loads sprite images from `Data-Acquisition/split_sprites/`~~
- ~~Reads type labels from `pokeapi_data/<id>_<name>/<name>.json` â†’ `types` field~~
- ~~Returns `(image_tensor, label)` pairs~~
- ~~Support single-label (Type 1 only) for now â€” add dual-type later~~
- ~~Fixed: RGBA â†’ RGB alpha composite on white, ImageNet normalization~~
- ~~Shared: `GEN_RANGES`, `get_generation`, `gen_stratified_split` live here for all files to import~~
- Run `python Classification/dataset.py` for sample output (index size, label distribution, one batch shape)

### ~~2. Baselines (`Classification/baselines.py`)~~ âœ…
- ~~Flatten images â†’ PCA â†’ decision tree, random forest, SVM~~
- ~~Quick to run, gives a comparison floor for the CNN~~
- ~~Scikit-learn is fine here~~
- Run `python Classification/baselines.py` â€” prints accuracy/F1/precision/recall for each model

### ~~3. CNN (`Classification/cnn_model.py`)~~ âœ…
- ~~Fine-tune ResNet18 (PyTorch + torchvision)~~
- ~~This is your main model â€” everything else is compared against it~~
- ~~Alpha composite sprites onto white background before passing to ResNet~~
- Run `python Classification/cnn_model.py` for a forward-pass sanity check (random input â†’ output shape)

### ~~4. Training script (`Classification/train.py`)~~ âœ…
- ~~Stratified split by generation â€” 70/15/15 train/val/test from each gen~~
- ~~WeightedRandomSampler â€” handles type imbalance~~
- ~~Log accuracy, F1, precision, recall per epoch~~
- ~~Save best checkpoint to `Classification/checkpoints/best.pt`~~
- Run `python Classification/train.py --epochs 1` to sanity check, then `--epochs 30` for full training

### ~~5. Evaluation (`Classification/evaluate.py`)~~ âœ…
- ~~Accuracy, F1 (macro + per-type), precision, recall, ROC-AUC~~
- ~~Confusion matrix by type â€” this is the most interesting output~~
- ~~Saves `confusion_matrix.npy`, `y_true.npy`, `y_pred.npy`, `y_probs.npy` to `Classification/results/` for Ajmain~~
- Run `python Classification/evaluate.py` â€” loads best checkpoint and prints full report

### 6. Ablations (after baseline works)
- **Grayscale run** â€” same model, strip color â†’ tests if shape alone is enough
- **Gen split** â€” train on gens 1-7, test on 8-9 â†’ tests generalization

---

## File Layout
```
Classification/
    dataset.py       â† shared module (index, split logic, transforms)
    baselines.py     â† run directly
    cnn_model.py     â† imported by train.py
    train.py         â† run directly
    evaluate.py      â† run directly
    checkpoints/     â† saved by train.py (gitignored)
```

---

## Key Numbers
- 1025 Pokemon, 18 types, up to 2 types each
- Start with Type 1 only (single-label, 18 classes) â€” simpler
- Add dual-type (multi-label) later if time permits

---

## Metrics To Report
- Accuracy
- Precision, Recall, F1 (macro-averaged)
- ROC-AUC
- Confusion matrix (heatmap by type)

---

## 7. HTML Visuals for Presentation (PRIORITY â€” needed tomorrow)

Each stage of the pipeline should produce a self-contained HTML file that can be opened in a browser for the presentation. Status:

| Stage | Script | HTML output | Done? |
|---|---|---|---|
| Data distribution | `run_analysis.py` | `generation_type_distribution.html` | âœ… |
| Baseline comparison | `baselines.py` | `Classification/results/baselines_comparison.html` | âœ… |
| Baseline confusion matrices | `baselines.py` | `Classification/results/baselines_confusion_matrices.html` | âœ… |
| Baseline mistake gallery | `baselines.py` | `Classification/results/mistakes_<Model>.html` | âœ… |
| CNN training curves | `train.py` | `Classification/results/training_curves.html` | âŒ needs building |
| CNN confusion matrix | `evaluate.py` | *(saved as .npy â€” needs HTML export)* | âŒ needs building |
| CNN mistake gallery | `evaluate.py` | `Classification/results/mistakes_CNN.html` | âœ… |
| Baseline vs CNN comparison | `evaluate.py` | *(not built)* | âŒ needs building |

**What to build before presentation:**
- `train.py` â€” save loss/accuracy/F1 per epoch to a plotly HTML (training curves)
- `evaluate.py` â€” export confusion matrix as interactive plotly heatmap HTML
- `evaluate.py` â€” add a bar chart comparing best baseline vs CNN on all metrics

---

## Notes
- Pixel sprites are small (~96x96 or smaller) â€” ResNet18 is fine, no need for anything bigger
- The greyscale ablation is important for the paper's argument about whether design is intentional
- CNN uses full RGB (3-channel) after alpha compositing sprites onto white â€” color IS part of what it learns
- Consider using all animation frames per Pokemon as separate training samples (~10x more data, helps rare types)
- Ask teammates if you need help with any of this

---

## Future Work Ideas (for presentation / paper)
- **More data / augmentation** â€” 688 training sprites is tiny; flips, color jitter, rotations, or adding shiny/back sprites could help significantly
- **Smaller model** â€” ResNet-18 is overkill for 96Ã-96 pixel art; a lightweight custom CNN would overfit less
- **Multi-label classification** â€” most PokÃ©mon have two types but we only predict one; a dual-output model would be more correct
- **Better RF features** â€” raw pixels aren't enough; try HSV histograms or edge/shape descriptors
- **Ensemble** â€” combine CNN confidence scores with RF predictions
- **Domain-appropriate pretraining** â€” ResNet-18 was trained on real photos, not pixel art; pretraining on anime/sprite datasets could help
- **Type-aware loss** â€” some types are semantically close (Ice/Water, Grass/Bug); use the in-game type relationship graph to inform the loss function
- **Attention maps** â€” visualize what part of the sprite the CNN actually looks at (Grad-CAM)
- **Generational analysis** â€” newer gen sprites are more detailed/3D vs Gen 1 pixel art; test if performance differs by generation
- **Dual-head model** â€” one output head for primary type, one for secondary type, trained jointly

