# Plan: Replace threshold with top-k in train.py val metric

## Context
Training currently uses a fixed 0.35 threshold to convert sigmoid outputs to predictions for computing val F1. This is misleading — it can predict 0, 1, 3+ types per Pokemon, making val F1 artificially low compared to evaluate.py which uses top-k (predict exactly as many types as the Pokemon actually has). This causes the val F1 during training to look much worse than the real test F1 (0.40 vs 0.80), making it hard to track true progress during training.

66/135 test Pokemon are single-type, so top-2 always would hurt them. Top-k is the correct approach.

## Change

**File:** `Classification/train.py`

**In `run_epoch`, replace lines 50-52:**
```python
preds = (torch.sigmoid(logits) > PRED_THRESHOLD).cpu().int().numpy()
all_preds.append(preds)
all_labels.append(labels.cpu().int().numpy())
```

**With:**
```python
probs = torch.sigmoid(logits).cpu().numpy()
labels_np = labels.cpu().int().numpy()
preds = np.zeros_like(probs, dtype=int)
for i in range(len(labels_np)):
    k = max(1, int(labels_np[i].sum()))
    top_k = np.argsort(probs[i])[-k:]
    preds[i, top_k] = 1
all_preds.append(preds)
all_labels.append(labels_np)
```

Also remove `PRED_THRESHOLD` from the import on line 14 since it's no longer used in train.py.

## Also Implement: Partial & Dual-Type Metrics

Add these to both train.py and evaluate.py:

- **Partial accuracy** - score per Pokemon: 1.0 if all predicted types correct, 0.5 if at least 1 correct, 0.0 if none correct. Average across all Pokemon. Applies to both single and dual-type.
- **Dual-type F1** - compute F1 only on the 69 dual-type Pokemon separately from single-type

## Verification
- Run `python Classification/train.py --epochs 1` and confirm val F1 is now in a reasonable range (should be closer to 0.80 than 0.40)
- No changes to evaluate.py, model, or checkpoint format — existing best.pt still works
