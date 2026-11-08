"""One dataset, all three methods, with progress printed. Use it to watch a single training and read the code.

    python run_one.py                       # Kumar's setting, run 0
    python run_one.py --mu 4 --windows 50   # another setting
"""
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import torch  # noqa: E402

from vdp.data import make_dataset  # noqa: E402
from vdp.evaluate import discovery_r2, judge  # noqa: E402
from vdp.least_squares import fit_least_squares  # noqa: E402
from vdp.neural_ode import fit_neural_ode  # noqa: E402

torch.set_num_threads(1)
ap = argparse.ArgumentParser()
ap.add_argument("--mu", type=float, default=1.5)
ap.add_argument("--sigma", type=float, default=0.1)
ap.add_argument("--n", type=int, default=100)
ap.add_argument("--run", type=int, default=0)
ap.add_argument("--windows", type=int, default=10)
ap.add_argument("--methods", default="ls,mlp,kan")
a = ap.parse_args()

t, clean, noisy = make_dataset("vdp", a.mu, a.sigma, a.n, a.run)
print(f"dataset: mu = {a.mu}, {a.n} points on [0, 10], noise sigma = {a.sigma}, run {a.run}")
print(f"first 3 noisy states: {noisy[:3].round(3).tolist()}")
scale = 6.5 if a.mu >= 3 else 2.5
for method in a.methods.split(","):
    print(f"\n=== {method.upper()} ===")
    if method == "ls":
        res = fit_least_squares(t, noisy)
    else:
        res = fit_neural_ode(t, noisy, method, windows=a.windows, scale=scale, seed=a.run, verbose=True)
    v = judge(res, a.mu, a.sigma)
    print(f"learned mu = {res.get('p_hat', float('nan')):.4f}   true mu = {a.mu}   error = {100 * v['rel_error']:.2f}%   "
          f"verdict = {v['failure']}   weights = {res['n_weights']}   time = {res['wall_s']:.1f} s   "
          f"Discovery R2 = {discovery_r2(res, 'vdp', a.mu):.3f}")
