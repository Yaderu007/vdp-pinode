# vdp-pinode

Recover the Van der Pol parameter µ from noisy measurements with three methods, on the same data:

| method | what it is | trainable numbers |
|---|---|---|
| `ls` | classical least squares: fit µ and the start state directly to the data with SciPy | 3 |
| `mlp` | physics-informed Neural ODE with an MLP 2 → 32 → 32 → 2 (tanh) | 1,218 + µ + window starts |
| `kan` | physics-informed Neural ODE with a KAN 2 → 10 → 2 (grid 5, cubic) | 400 + µ + window starts |

Project brief: UgS research project, supervisor Omar Elshinawy. Method references: Kumar (multiple-shooting Neural ODE),
Raissi et al. 2019 (physics-informed loss), Chen et al. 2018 (Neural ODE, `torchdiffeq`), Liu et al. 2024 (KAN).

## Files: one per step of the project map

| map step | file | what it does |
|---|---|---|
| all fixed numbers | `vdp/settings.py` | every setting with the reason for its value |
| 2 system & state | `vdp/system.py` | Van der Pol (and Duffing) as `x' = v, v' = ...` |
| 3 data | `vdp/data.py` | exact solution → 100 samples → noise, with a seed per setting |
| 5–7 scaling, MLP, KAN | `vdp/networks.py` | `RateField` divides the state by 2.5 and calls the MLP or KAN |
| 4, 8–11 model & training | `vdp/neural_ode.py` | solver, shooting windows, the three losses, Adam then L-BFGS |
| 12 judging | `vdp/evaluate.py` | relative error, success rule, failure type, Discovery R² |
| 15 baseline | `vdp/least_squares.py` | classical least squares, no network |
| run one dataset | `run_one.py` | trains all three methods on one dataset and prints progress |
| run experiments | `run_experiment.py` | plans (`kumar`, `grid`, ...), parallel, resumable, writes `results/*.csv` |
| tables | `summarize.py` | one row per setting and method |
| plots | `make_figures.py` | error per run, µ during training |

Project explained step by step: [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)

Project explained step by step: [docs/PROJECT_MAP.md](docs/PROJECT_MAP.md)

Suggested reading order: `settings.py` → `system.py` → `data.py` → `networks.py` → `neural_ode.py` →
`least_squares.py` → `evaluate.py` → `run_one.py`.

## Setup (once)

```bash
# 1. the project environment (already exists on your PC)
conda activate vdp                        # prompt shows (vdp); python --version -> 3.13

# 2. general libraries
pip install numpy scipy pandas matplotlib tqdm

# 3. the official repositories, installed from your own clones at the exact versions used
mkdir -p ~/vdp_project/repos && cd ~/vdp_project/repos
[ -d torchdiffeq ]   || git clone https://github.com/rtqichen/torchdiffeq.git
[ -d efficient-kan ] || git clone https://github.com/Blealtan/efficient-kan.git
git -C torchdiffeq checkout 657943a
git -C efficient-kan checkout 7b6ce1c
pip install --no-deps -e ./torchdiffeq
pip install --no-deps -e ./efficient-kan

# 4. this repository
cd ~/vdp_project
git clone https://github.com/Yaderu007/vdp-pinode.git
cd vdp-pinode
python check_setup.py                     # must end with "all good"
```

Always start Python from inside `vdp-pinode/` (not from `repos/`): a folder named `torchdiffeq` in the current
directory would hide the installed package.

## Running

| command | what | time on one CPU core |
|---|---|---|
| `python run_one.py` | Kumar's setting, run 0, all three methods, with progress | ~3 min |
| `python run_experiment.py --plan check` | same dataset through the experiment runner | ~2 min |
| `python run_experiment.py --plan kumar --runs 50` | Kumar's setting, 50 repeats | ~2.5 h CPU time |
| `python run_experiment.py --plan grid --runs 10` | µ {0.5, 1.5, 4} × noise {0.01, 0.1, 0.3} | ~1.5 h CPU time |
| `python run_experiment.py --plan density --runs 5` | µ {0.5, 1.5, 4} × points {50, 100, 400} | ~1.5 h CPU time |
| `python run_experiment.py --plan robustness --runs 5` | other start states, Laplace noise, small MLP | ~20 min CPU time |
| `python run_experiment.py --plan duffing --runs 10` | Duffing oscillator, recover β | ~1 h CPU time |

CPU time is divided by the number of workers (`--workers`, default: all cores but one).
Ctrl+C is safe; the same command continues where it stopped. Then:

```bash
python summarize.py --plan kumar          # table -> results/kumar_summary.csv
python make_figures.py --plan kumar       # plots -> figures/
```

### Why the CPU and not the GPU
The networks are tiny (400–1,218 numbers) and the solver calls them about 14 times per training step on at most
100 states. On a GPU each call has a fixed start-up cost that is larger than the work itself, so it is slower.
On the CPU, one run per core in parallel is fastest, and every method runs on the same device.

## Output columns (`results/<plan>.csv`)

`run_id` unique name of the run · `method` ls / mlp / kan / mlp16 · `p_true` true µ · `sigma` noise · `n_points` ·
`start_state` 0 = (2, 0), 1 = (0.5, 0.5), 2 = (−1, 1.5) · `noise` gaussian / laplace · `windows` shooting windows
(0 for ls) · `input_scale` · `run` repeat number (seed) · `p_hat` learned µ · `rel_error` |p_hat − p_true| / p_true ·
`success` · `failure` none / no_fit / wrong_parameter / crash · `data_mse` mean squared distance to the data ·
`physics_mse` network vs formula · `discovery_r2` learned vs true µ(1 − x²)v on a 100 × 100 grid ·
`trajectory_r2` the same only at the states the motion visits · `n_weights` · `wall_s` seconds.

`results/<plan>_traces.csv`: every 100 Adam steps: `step`, `data_loss`, `physics_loss`, `p` (current µ).
