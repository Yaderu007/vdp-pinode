"""Map step 15: the classical least-squares baseline (no neural network).

The equation is fully known except mu, so we can fit mu directly:
    1. guess mu and the start state (x0, v0)          (start: mu = 1.0, start state = first data point)
    2. solve Van der Pol with SciPy at the data times
    3. residuals = simulated states - measured states  (200 numbers)
    4. SciPy's least_squares changes (mu, x0, v0) to make the sum of squared residuals smallest.
The start state is fitted too, because the first measurement is itself noisy.
"""
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from . import settings as S
from .system import SYSTEMS, rate_numpy


def fit_least_squares(t, noisy, eq="vdp", p_guess=None):
    p0 = float(p_guess if p_guess is not None else SYSTEMS[eq]["p_guess"])

    def residuals(theta):
        p, x0, v0 = theta
        sol = solve_ivp(rate_numpy(eq, p), (t[0], t[-1]), [x0, v0], t_eval=t, method="DOP853", rtol=1e-8, atol=1e-10)
        if not sol.success or sol.y.shape[1] != len(t):  # solver blew up for this guess: return a large misfit
            return np.full(noisy.size, 1e3)
        return (sol.y.T - noisy).ravel()

    t0 = time.perf_counter()
    try:
        fit = least_squares(residuals, x0=[p0, noisy[0, 0], noisy[0, 1]], method="trf", x_scale="jac")
        status, note = ("ok", "") if np.all(np.isfinite(fit.x)) else ("crash", "non-finite result")
    except Exception as err:
        fit, status, note = None, "crash", f"{type(err).__name__}: {err}"[:200]
    out = dict(method="ls", status=status, note=note, wall_s=time.perf_counter() - t0, n_weights=0,
               windows=0, input_scale=0.0, trace=[])
    if status == "ok":
        r = residuals(fit.x).reshape(-1, 2)
        out.update(p_hat=float(fit.x[0]), data_mse=float((r ** 2).mean()), physics_mse=0.0,
                   prediction=noisy + r, start_fit=fit.x[1:].tolist(), solver_calls=int(fit.nfev))
    return out
