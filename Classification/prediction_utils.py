"""Shared prediction and metric helpers for multi-label type classification."""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

DEFAULT_GAP_THRESHOLD = 0.25


def predict_from_probabilities(probs, gap_threshold: float = DEFAULT_GAP_THRESHOLD):
    """Predict top-1 plus top-2 when the probability gap is small.

    The project currently uses this rule for live train/validation metrics and
    test evaluation: every sample gets its highest-scoring type, and gets a
    second type only when the top two probabilities differ by less than
    ``gap_threshold``.
    """
    if isinstance(probs, torch.Tensor):
        return _predict_tensor(probs, gap_threshold)
    return _predict_numpy(np.asarray(probs), gap_threshold)


def _validate_probabilities(probs) -> None:
    if len(probs.shape) != 2 or probs.shape[1] < 2:
        raise ValueError("Expected probabilities with shape (n_samples, n_classes >= 2)")


def _predict_tensor(probs: torch.Tensor, gap_threshold: float) -> torch.Tensor:
    _validate_probabilities(probs)
    sorted_probs, sorted_idx = torch.sort(probs, dim=1, descending=True)
    preds = torch.zeros_like(probs, dtype=torch.int)
    rows = torch.arange(probs.shape[0], device=probs.device)

    preds[rows, sorted_idx[:, 0]] = 1
    second_type = (sorted_probs[:, 0] - sorted_probs[:, 1]) < gap_threshold
    preds[rows[second_type], sorted_idx[second_type, 1]] = 1
    return preds


def _predict_numpy(probs: np.ndarray, gap_threshold: float) -> np.ndarray:
    _validate_probabilities(probs)
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]
    rows = np.arange(probs.shape[0])
    preds = np.zeros_like(probs, dtype=int)

    preds[rows, sorted_idx[:, 0]] = 1
    second_type = (probs[rows, sorted_idx[:, 0]] - probs[rows, sorted_idx[:, 1]]) < gap_threshold
    preds[rows[second_type], sorted_idx[second_type, 1]] = 1
    return preds


def collect_gap_predictions(
    model,
    loader,
    device,
    gap_threshold: float = DEFAULT_GAP_THRESHOLD,
    desc: str = "Evaluating",
    show_progress: bool = True,
):
    """Collect labels, gap-rule predictions, and sigmoid probabilities."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    batches = tqdm(loader, desc=desc) if show_progress else loader

    with torch.no_grad():
        for imgs, labels in batches:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu()
            preds = predict_from_probabilities(probs, gap_threshold=gap_threshold)

            all_labels.append(labels.int().numpy())
            all_preds.append(preds.numpy())
            all_probs.append(probs.numpy())

    return (
        np.vstack(all_labels),
        np.vstack(all_preds),
        np.vstack(all_probs),
    )


def multilabel_metrics(y_true, y_pred) -> dict[str, float]:
    """Return the macro multi-label metrics used throughout the project."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
    }
