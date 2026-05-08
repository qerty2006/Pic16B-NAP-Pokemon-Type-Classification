import sys
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

csv_file = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "log.csv"
df = pd.read_csv(csv_file)

train = df[df["phase"] == "train"].set_index("epoch")
val   = df[df["phase"] == "val"].set_index("epoch")

metrics = [
    ("loss",      "Loss"),
    ("accuracy",  "Accuracy"),
    ("f1",        "F1 (macro)"),
    ("precision", "Precision (macro)"),
    ("recall",    "Recall (macro)"),
]

fig = make_subplots(rows=2, cols=3, subplot_titles=[m[1] for m in metrics])

positions = [(1,1),(1,2),(1,3),(2,1),(2,2)]

for (col, title), (row, c) in zip(metrics, positions):
    fig.add_trace(go.Scatter(x=train.index, y=train[col], name="train",
                             line=dict(width=2), legendgroup="train",
                             showlegend=(row==1 and c==1)), row=row, col=c)
    fig.add_trace(go.Scatter(x=val.index, y=val[col], name="val",
                             line=dict(width=2, dash="dash"), legendgroup="val",
                             showlegend=(row==1 and c==1)), row=row, col=c)

fig.update_layout(
    title_text="Training Curves — EfficientNet-B0 Pokémon Type Classifier",
    height=600, width=1100,
)

out = Path(csv_file).stem + "_curves.html"
fig.write_html(out)
print(f"Saved {out}")
