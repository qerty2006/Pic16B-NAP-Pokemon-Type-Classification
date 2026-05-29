# Post-Refactor Error Scan & Cleanup

## Context

`codex` recently refactored this codebase (commit `eb44b9b`, "experimental push", followed by
doc-only commits `b103056` and `cd5791b`). The refactor consolidated the canonical workflow into
`Classification/`, extracted shared prediction logic into `prediction.py`, removed duplicate
functions (`run_epoch2`, `collect_predictions2`), dropped the old `Visualizer`/`Patrick` imports,
and archived experimental code. The goal is a **static logic scan** to catch any errors the
refactor introduced (no model training / full runs yet), plus a `TODO.md` reorganization since
several items are now implemented.

## Scan Result (headline)

**The refactor is structurally clean.** Every high-risk change flagged during exploration was
propagated consistently:
- No remaining references to `run_epoch2`, `collect_predictions2`, the old `n_types` signature,
  `from Visualizer`, `save_all_mistake_examples_with_probs`, or the `Patrick/` path anywhere in
  active code (only in `README.md` notes and `archive/`, which is expected).
- All cross-module imports resolve: `train.py`, `evaluate.py`, `pipeline.py`, `baselines.py`,
  `generate_report.py` all import names that still exist in `dataset.py` / `prediction.py` /
  `cnn_model.py` with matching signatures.
- The 5 unit tests in `tests/test_core_logic.py` are logically correct and should pass.

What remains are minor quality issues, not crashes.

## Findings

| # | Severity | Location | Issue | Disposition |
|---|----------|----------|-------|-------------|
| A | Low | `Classification/dataset.py:174` | `gen_gen_split` uses unsafe `int(folder_name.split("-")[0])` instead of the safe `parse_folder_id`; would crash on a non-numeric folder. `gen_stratified_split` already uses the safe helper. | **FIX (safe)** |
| C | Low | `Classification/dataset.py:43-51`, `:63-72` | `TRAIN_TRANSFORM` & `GRAYSCALE_TRAIN_TRANSFORM` apply `RandomHorizontalFlip` twice (once before `ToTensor`, once after). Redundant — net flip probability is unchanged. | **FIX (safe)** |
| B | Low | `Classification/dataset.py:132`, `Classification/evaluate.py:58` | `get_generation(pokemon_id)` is called without `folder_name`, so regional forms (alola/galar/etc.) are bucketed by base-ID generation. Inconsistent with `gen_gen_split`, which passes it. Changes stratification buckets and per-gen reporting → **alters splits/results**. | **FLAG (behavioral)** |
| D | Medium | `Classification/dataset.py:242` | `_fetch_entry` returns only `frames[:1]` — one sprite frame per folder, despite multi-frame folders. May be intentional (dedup near-identical frames), but the split test implies multi-frame support. Using all frames would change dataset size & every result. | **FLAG (confirm intent)** |
| E | Low | `Classification/pipeline.py:259` | `save_pipeline_gallery` renders **all** mistakes uncapped; `evaluate.py` caps at 30. Large test sets → very large HTML. Possibly an intentional full-diagnostic. | **FLAG (observation)** |
| F | Low | `Classification/ViT.py` | Not imported by the active workflow (only `archive/small_classif/` imports `ViT`). Likely dead/standalone. `visualize_cnn.py` is standalone too but its imports are consistent and valid. | **FLAG (keep vs archive)** |

## Plan of Action

### 1. Apply safe fixes (no behavior change on valid data)
- **A — `Classification/dataset.py:172-175`:** in `gen_gen_split`, replace
  `pokemon_id = int(folder_name.split("-")[0])` with `pokemon_id = parse_folder_id(folder_name)`
  and `continue` when it is `None` (mirrors `gen_stratified_split`).
- **C — `Classification/dataset.py:43-51` & `:63-72`:**
  remove the duplicate `transforms.RandomHorizontalFlip(p=0.5)` that sits after `ToTensor()` in
  both train transforms; keep the single flip and move the "Swaps facing direction" comment onto it.
  (ColorJitter ordering left as-is to avoid any numeric change — noted as optional future cleanup.)

### 2. Flagged items requiring a decision (B, D, E, F)
Behavioral — not changed without sign-off. B and D are the only ones that affect model results.

### 3. Reorganize `TODO.md` into sections
Restructure the flat list into **Done (code-verified) / In progress / Remaining**, based on what the
code shows, without overclaiming writeup/analysis work:
- **Done (implemented in code):** port canonical workflow to `Classification/` (refactor); repo
  cleanup (archive/, tests added); grayscale *support* (`GRAYSCALE_*` transforms + `--grayscale`
  flag + grayscale gallery); neuron / per-color-channel visualization (`Classification/visualize_cnn.py`);
  data collection & analysis scripts present.
- **In progress:** finalize main model scoring for the paper (metrics pipeline exists in
  `Classification/evaluate.py` / `Classification/generate_report.py`).
- **Remaining:** single-type vs dual-type accuracy metric (not found in `evaluate.py`); all
  PRESENTATION writeups (how/why models, data analysis narrative) — can't be verified from code, kept as-is.
- Preserve the `delete cache -> dataset -> train -> evaluate -> visualizer` workflow note as a
  "run order" reminder.

## Verification (no training run)
1. `python -m unittest discover -s tests` — the 5 core-logic tests should still pass after edits A/C.
2. Import smoke test (catches syntax/import breakage without running models, since `main()` is
   guarded by `__main__`):
   `python -c "import sys; sys.path.insert(0,'Classification'); import dataset, cnn_model, prediction, train, evaluate, pipeline, baselines, generate_report, visualize_cnn"`
3. Optional: `python -m py_compile Classification/*.py tests/*.py` for a pure syntax pass.
4. Confirm `TODO.md` renders as intended and no done-claim lacks code evidence.

No full `train.py` / `pipeline.py --mode all` run is part of this task (scan logic only).
