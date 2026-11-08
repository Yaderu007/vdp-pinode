"""Run an experiment plan: every dataset x every method x every run, in parallel on the CPU. Resumable.

    python run_experiment.py --plan check                 # 1 dataset, MLP + KAN + least squares (~3 min)
    python run_experiment.py --plan kumar --runs 50       # Kumar's setting, 50 repeats
    python run_experiment.py --plan grid --runs 10        # mu {0.5, 1.5, 4} x noise {0.01, 0.1, 0.3}
    python run_experiment.py --plan density --runs 5      # mu {0.5, 1.5, 4} x points {50, 100, 400}
    python run_experiment.py --plan robustness --runs 5   # other start states, Laplace noise, small MLP
    python run_experiment.py --plan duffing --runs 10     # second system of the brief (recover beta)

Output (one row per run):  results/<plan>.csv          training traces: results/<plan>_traces.csv
Stopping with Ctrl+C is safe: run the same command again and finished runs are skipped.
"""
import argparse
import csv
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # everything on the CPU: same device for every method (see README)
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

NETS = ["mlp", "kan"]


def cell(eq="vdp", p=1.5, sigma=0.1, n=100, start=0, noise="gaussian", windows=50, methods=("mlp", "kan", "ls")):
    scale = 6.5 if (eq == "vdp" and p >= 3) else 2.5    # map step 5: at mu = 4 the velocity leaves the 2.5 range
    return dict(eq=eq, p_true=p, sigma=sigma, n=n, start=start, noise=noise, windows=windows, scale=scale,
                methods=list(methods))


def plan_cells(plan):
    if plan == "check":
        return [cell(windows=10)]
    if plan == "kumar":                                   # Kumar's method: 10 windows of 10 points
        return [cell(windows=10)]
    if plan == "grid":                                    # 50 windows of 2 points (needed for mu = 4)
        return [cell(p=p, sigma=s) for p in (0.5, 1.5, 4.0) for s in (0.01, 0.1, 0.3)]
    if plan == "density":
        return [cell(p=p, n=n, windows=25 if n == 50 else 50) for p in (0.5, 1.5, 4.0) for n in (50, 100, 400)]
    if plan == "robustness":
        return [cell(start=1), cell(start=2), cell(noise="laplace"), cell(methods=("mlp16",))]
    if plan == "duffing":
        return [cell(eq="duffing", p=0.3, sigma=s, windows=10) for s in (0.01, 0.1, 0.3)]
    raise SystemExit(f"unknown plan '{plan}'")


def run_id(plan, c, method, run):
    return (f"{plan}|{c['eq']}|p{c['p_true']:g}|s{c['sigma']:g}|n{c['n']}|start{c['start']}|{c['noise']}|"
            f"{method}|w{c['windows']}|run{run}")


def job(args):
    """Runs in a worker process: make the dataset, fit one method, judge it."""
    import torch
    torch.set_num_threads(1)
    from vdp.data import make_dataset
    from vdp.evaluate import discovery_r2, judge
    from vdp.least_squares import fit_least_squares
    from vdp.neural_ode import fit_neural_ode
    plan, c, method, run = args
    t, clean, noisy = make_dataset(c["eq"], c["p_true"], c["sigma"], c["n"], run, c["start"], c["noise"])
    if method == "ls":
        res = fit_least_squares(t, noisy, c["eq"])
    else:
        res = fit_neural_ode(t, noisy, method, c["eq"], c["windows"], c["scale"], seed=run)
    verdict = judge(res, c["p_true"], c["sigma"])
    row = dict(run_id=run_id(plan, c, method, run), plan=plan, system=c["eq"], method=method,
               p_true=c["p_true"], sigma=c["sigma"], n_points=c["n"], start_state=c["start"], noise=c["noise"],
               windows=res["windows"], input_scale=res["input_scale"], run=run, status=res["status"],
               p_hat=res.get("p_hat", float("nan")), rel_error=verdict["rel_error"], success=verdict["success"],
               failure=verdict["failure"], data_mse=res.get("data_mse", float("nan")),
               physics_mse=res.get("physics_mse", float("nan")),
               discovery_r2=discovery_r2(res, c["eq"], c["p_true"]),
               trajectory_r2=discovery_r2(res, c["eq"], c["p_true"], states=clean),
               n_weights=res["n_weights"], wall_s=round(res["wall_s"], 2), note=res["note"])
    traces = [dict(run_id=row["run_id"], **tr) for tr in res["trace"]]
    return row, traces


def append(path, rows):
    if not rows:
        return
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        if new:
            w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--runs", type=int, default=None, help="repeats per dataset setting (default depends on plan)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--methods", default=None, help="e.g. mlp,ls to run only some methods")
    a = ap.parse_args()
    runs = a.runs or dict(check=1, kumar=50, grid=10, density=5, robustness=5, duffing=10)[a.plan]
    RESULTS.mkdir(exist_ok=True)
    out, out_tr = RESULTS / f"{a.plan}.csv", RESULTS / f"{a.plan}_traces.csv"
    done = set()
    if out.exists():
        with open(out) as f:
            done = {r["run_id"] for r in csv.DictReader(f)}
    cells = plan_cells(a.plan)
    only = a.methods.split(",") if a.methods else None
    # run 0 of every setting first, then run 1, ...: stopping early leaves the same number of runs everywhere
    jobs = [(a.plan, c, m, r) for r in range(runs) for c in cells for m in c["methods"]
            if (only is None or m in only) and run_id(a.plan, c, m, r) not in done]
    print(f"plan '{a.plan}': {len(cells)} dataset settings, {len(jobs)} runs to do ({len(done)} already done), "
          f"{a.workers} parallel workers", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        futures = [ex.submit(job, j) for j in jobs]
        for k, fut in enumerate(as_completed(futures), 1):
            row, traces = fut.result()
            append(out, [row]); append(out_tr, traces)
            print(f"[{k}/{len(jobs)} | {time.time() - t0:6.0f} s] {row['method']:5s} mu={row['p_true']:g} "
                  f"sigma={row['sigma']:g} n={row['n_points']} start={row['start_state']} {row['noise']} "
                  f"run={row['run']}: {row['failure']:15s} estimate={row['p_hat']:.4f} "
                  f"error={100 * row['rel_error']:.2f}%  ({row['wall_s']} s)", flush=True)
    print(f"done. results in {out}. Next: python summarize.py --plan {a.plan}")
