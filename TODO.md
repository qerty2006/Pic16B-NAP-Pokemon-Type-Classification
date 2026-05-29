# TODO

## Done (implemented in code)
- [x] Port canonical workflow to `Classification/` (codex refactor) — Ajmain
- [x] Clean up repo: experimental code moved to `archive/`, unit tests added in `tests/` — ALL
- [x] Grayscale support: `GRAYSCALE_*` transforms + `--grayscale` pipeline flag + grayscale mistake gallery — Nish
- [x] Neuron / per-color-channel visualization, colored outlines per channel (`Classification/visualize_cnn.py`) — Nish
- [x] Data collection (`Data-Acquisition/`) and data analysis (`Data-Analysis/`) scripts — Nish / Ajmain

## In progress
- [ ] Finalize main model scoring for the paper — metrics pipeline exists in `Classification/evaluate.py` + `generate_report.py`; lock in final numbers — Ajmain
- [ ] Grayscale study writeup — run color vs grayscale comparison and document results — Nish

## Remaining
- [ ] Accuracy metric split by single-type vs dual-type Pokemon (not yet in `evaluate.py`) — Ajmain
- [ ] PRESENTATION — How we trained the model, why we used certain models — Patrick
- [ ] PRESENTATION — Data analysis narrative — Ajmain
- [ ] PRESENTATION — Data collection + grayscale study + neuron visualization walkthrough — Nish

## Run order (reminder)
delete cache -> dataset -> train -> evaluate -> visualizer
