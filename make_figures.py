"""Plots from your own results.  python make_figures.py --plan kumar   (or grid, density, ...)  ->  figures/<plan>_*.png"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
COLOR = {"ls": "#1D7A4E", "mlp": "#2F5FB8", "kan": "#C2571A", "mlp16": "#7A4CC2"}
NAME = {"ls": "least squares", "mlp": "MLP", "kan": "KAN", "mlp16": "small MLP"}
plt.rcParams.update({"font.size": 13, "axes.spines.top": False, "axes.spines.right": False})

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
a = ap.parse_args()
df = pd.read_csv(HERE / "results" / f"{a.plan}.csv")
(HERE / "figures").mkdir(exist_ok=True)
methods = [m for m in ["ls", "mlp", "kan", "mlp16"] if m in set(df.method)]

# 1. error of every run, per dataset setting and method (one dot per run, bar = median)
df["setting"] = df.apply(lambda r: f"µ={r.p_true:g} σ={r.sigma:g} n={r.n_points}"
                         + (f" start{r.start_state}" if r.start_state else "") + (" laplace" if r.noise != "gaussian" else ""), axis=1)
settings = list(dict.fromkeys(df.sort_values(["p_true", "sigma", "n_points", "start_state"]).setting))
fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(settings) * len(methods) / 2), 5.5), dpi=150)
w = 0.8 / len(methods)
for j, m in enumerate(methods):
    for i, s in enumerate(settings):
        e = 100 * df[(df.method == m) & (df.setting == s)].rel_error.dropna().values
        if len(e) == 0:
            continue
        x = i + (j - (len(methods) - 1) / 2) * w
        ax.scatter(x + np.random.default_rng(i).uniform(-w / 4, w / 4, len(e)), np.clip(e, 1e-3, None), s=18,
                   color=COLOR[m], alpha=0.6, label=NAME[m] if i == 0 else None)
        ax.plot([x - w / 3, x + w / 3], [np.median(e)] * 2, color="black", lw=2)
ax.set_yscale("log"); ax.axhline(10, color="#B23A2A", ls="--", lw=1); ax.text(len(settings) - 0.5, 11, "10% = failure", color="#B23A2A", ha="right")
ax.set_xticks(range(len(settings))); ax.set_xticklabels(settings, rotation=30, ha="right")
ax.set_ylabel("error in the parameter (%), log scale"); ax.legend(frameon=False)
ax.set_title(f"{a.plan}: one dot per run, black bar = median", loc="left")
fig.tight_layout(); fig.savefig(HERE / "figures" / f"{a.plan}_errors.png"); plt.close(fig)

# 2. how mu moves during Adam training (median over runs), from the traces
tr_path = HERE / "results" / f"{a.plan}_traces.csv"
if tr_path.exists():
    tr = pd.read_csv(tr_path).merge(df[["run_id", "method", "p_true"]], on="run_id")
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for m in [m for m in methods if m != "ls"]:
        g = tr[tr.method == m].assign(ratio=lambda d: d.p / d.p_true).groupby("step").ratio
        ax.plot(g.median().index, g.median().values, color=COLOR[m], lw=2.5, label=NAME[m])
        ax.fill_between(g.median().index, g.quantile(0.1).values, g.quantile(0.9).values, color=COLOR[m], alpha=0.15)
    ax.axhline(1, color="black", ls="--", lw=1)
    ax.set_xlabel("Adam step"); ax.set_ylabel("learned / true parameter (1 = correct)")
    ax.set_title("during training (median, band = 10-90% of runs); L-BFGS follows", loc="left"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(HERE / "figures" / f"{a.plan}_training.png"); plt.close(fig)
print("figures written to", HERE / "figures")
