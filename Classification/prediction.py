"""Shared prediction helpers for multi-label Pokemon type outputs."""

from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

GAP_THRESHOLD = 0.25


def predict_from_probs(probs, gap_threshold: float = GAP_THRESHOLD) -> np.ndarray:
    """Convert class probabilities to multi-hot predictions.

    The model always predicts the highest-probability type. It predicts the
    second-highest type only when it is within ``gap_threshold`` of the top type.
    """
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError("probs must be a 2D array of shape (n_samples, n_classes)")
    if probs.shape[1] == 0:
        raise ValueError("probs must contain at least one class")

    preds = np.zeros_like(probs, dtype=int)
    sorted_idx = np.argsort(probs, axis=1)[:, ::-1]
    rows = np.arange(probs.shape[0])

    preds[rows, sorted_idx[:, 0]] = 1
    if probs.shape[1] > 1:
        top = probs[rows, sorted_idx[:, 0]]
        second = probs[rows, sorted_idx[:, 1]]
        include_second = (top - second) < gap_threshold
        preds[rows[include_second], sorted_idx[include_second, 1]] = 1

    return preds


def collect_predictions(
    model,
    loader,
    device,
    gap_threshold: float = GAP_THRESHOLD,
    desc: str = "Evaluating",
):
    """Run model inference and collect labels, gap-threshold predictions, and probabilities."""
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=desc):
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs)).cpu().numpy()
            preds = predict_from_probs(probs, gap_threshold=gap_threshold)

            all_labels.append(labels.int().numpy())
            all_preds.append(preds)
            all_probs.append(probs)

    return (
        np.vstack(all_labels),
        np.vstack(all_preds),
        np.vstack(all_probs),
    )
