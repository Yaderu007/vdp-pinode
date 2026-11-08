"""Turn results/<plan>.csv into a table: one row per dataset setting and method.

    python summarize.py --plan kumar
Writes results/<plan>_summary.csv and prints the table.
"""
import argparse
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    a = ap.parse_args()
    df = pd.read_csv(HERE / "results" / f"{a.plan}.csv")
    keys = ["system", "p_true", "sigma", "n_points", "start_state", "noise", "method"]
    g = df.groupby(keys, sort=True)
    summary = pd.DataFrame({
        "runs": g.size(),
        "successes": g["success"].sum(),
        "median_error_%": g["rel_error"].median() * 100,
        "mean_estimate": g["p_hat"].mean(),
        "sd_estimate": g["p_hat"].std(),
        "worst_error_%": g["rel_error"].max() * 100,
        "median_discovery_R2": g["discovery_r2"].median(),
        "median_trajectory_R2": g["trajectory_r2"].median(),
        "median_seconds": g["wall_s"].median(),
        "weights": g["n_weights"].first(),
    }).round(4).reset_index()
    summary.to_csv(HERE / "results" / f"{a.plan}_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(summary.to_string(index=False))
