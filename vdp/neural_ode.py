"""Map steps 4 and 8-11: the physics-informed Neural ODE and its training.

    state --network--> rate --ODE solver--> trajectory --> compare with the data dots   (data loss)
                       rate <-> Van der Pol formula with trainable mu                    (physics loss)

Multiple shooting (step 9): the 100 points are cut into windows. Each window starts from its own trainable start
state (first set to the window's first data point) and the solver only runs inside the window. A continuity loss
makes the end of each window meet the start of the next one.
"""
import time

import numpy as np
import torch
from torchdiffeq import odeint

from . import settings as S
from .networks import RateField, count_weights, make_network
from .system import SYSTEMS, rate_torch

torch.set_default_dtype(torch.float64)   # double precision, as in the reference codes


def fit_neural_ode(t, noisy, network="mlp", eq="vdp", windows=10, scale=S.INPUT_SCALE, seed=0,
                   adam_steps=S.ADAM_STEPS, lbfgs_iters=S.LBFGS_ITERS, lr=S.ADAM_LR, physics_weight=S.PHYSICS_WEIGHT,
                   p_guess=None, verbose=False):
    """Train one model on one dataset. Returns a dict with the learned parameter, losses, time and a training trace."""
    n = len(t)
    m = n // windows                                     # data points per window
    assert m * windows == n, "the number of points must be divisible by the number of windows"
    z = torch.tensor(noisy)
    Z = z.reshape(windows, m, 2)                         # data, window by window
    h = float(t[1] - t[0])
    t_window = torch.arange(m + 1) * h                   # times inside one window, up to the next window's start

    torch.manual_seed(seed)                              # the random start values of the weights
    field = RateField(make_network(network), scale)
    starts = torch.nn.Parameter(Z[:, 0, :].clone())      # trainable start state of every window
    p = torch.nn.Parameter(torch.tensor(float(p_guess if p_guess is not None else SYSTEMS[eq]["p_guess"])))
    trainable = list(field.parameters()) + [starts, p]

    def losses():
        # all windows are solved together in one call: Y has shape (time, window, 2)
        Y = odeint(field, starts, t_window, method=S.SOLVER, rtol=S.RTOL, atol=S.ATOL).permute(1, 0, 2)
        pred = Y[:, :m, :]                                                   # predictions at the data times
        data = ((pred - Z) ** 2).mean()                                      # step 10: data loss
        continuity = ((Y[:-1, m, :] - starts[1:]) ** 2).sum(-1).mean()       # window end = next window start
        physics = ((field(None, pred) - rate_torch(eq, pred, p)) ** 2).mean()  # network rate vs formula rate
        return data, continuity, physics, pred.reshape(n, 2)

    trace = []
    t0 = time.perf_counter()
    status, note = "ok", ""
    try:
        # ---- stage 1: Adam ---------------------------------------------------------------------------
        opt = torch.optim.Adam(trainable, lr=lr)
        for step in range(adam_steps):
            opt.zero_grad()
            data, cont, phys, _ = losses()
            loss = data + cont + physics_weight * phys
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss became infinite at Adam step {step}")
            loss.backward()                              # gradients of the loss for every trainable number
            opt.step()                                   # nudge every number a little
            if step % S.TRACE_EVERY == 0:
                trace.append(dict(step=step, data_loss=data.item(), physics_loss=phys.item(), p=p.item()))
                if verbose:
                    print(f"  Adam step {step:5d}   data loss {data.item():.2e}   physics loss {phys.item():.2e}   "
                          f"{SYSTEMS[eq]['name']} = {p.item():.4f}", flush=True)
        # ---- stage 2: L-BFGS ------------------------------------------------------------------------
        if lbfgs_iters > 0:
            lbfgs = torch.optim.LBFGS(trainable, lr=1.0, max_iter=lbfgs_iters, history_size=50,
                                      tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

            def closure():
                lbfgs.zero_grad()
                d, c, ph, _ = losses()
                total = d + c + physics_weight * ph
                total.backward()
                return total
            lbfgs.step(closure)
        data, cont, phys, pred = losses()
        if verbose:
            print(f"  after L-BFGS        data loss {data.item():.2e}   physics loss {phys.item():.2e}   "
                  f"{SYSTEMS[eq]['name']} = {p.item():.4f}", flush=True)
        if not (torch.isfinite(data) and torch.isfinite(p)):
            raise FloatingPointError("result is not finite")
    except Exception as err:                             # a crash counts as a failed run, never a skipped one
        status, note = "crash", f"{type(err).__name__}: {err}"[:200]
    out = dict(method=network, status=status, note=note, wall_s=time.perf_counter() - t0,
               n_weights=count_weights(field.net), windows=windows, input_scale=scale, trace=trace)
    if status == "ok":
        out.update(p_hat=p.item(), data_mse=((pred.detach() - z) ** 2).mean().item(), physics_mse=phys.item(),
                   prediction=pred.detach().numpy())
    out["field"] = field                                 # kept in memory for plots and Discovery R^2
    return out
