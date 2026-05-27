import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from pathlib import Path

results = Path(__file__).parent / "results"
y_true = np.load(results / "y_true.npy")
y_pred = np.load(results / "y_pred.npy")

print(f"F1:        {f1_score(y_true, y_pred, average='macro', zero_division=0)*100:.1f}%")
print(f"Accuracy:  {accuracy_score(y_true, y_pred)*100:.1f}%")
print(f"Precision: {precision_score(y_true, y_pred, average='macro', zero_division=0)*100:.1f}%")
print(f"Recall:    {recall_score(y_true, y_pred, average='macro', zero_division=0)*100:.1f}%")