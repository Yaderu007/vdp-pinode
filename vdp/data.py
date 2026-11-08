"""Map step 3: making the data.

1. Solve the equation very accurately (SciPy, DOP853, tolerance 1e-10) -> the exact motion.
2. Keep n equally spaced moments on [0, T_END].
3. Add random noise with standard deviation sigma to x and v.
The random numbers come from a seed built from the settings, so the same settings always give the same data.
"""
import numpy as np
from scipy.integrate import solve_ivp

from . import settings as S
from .system import rate_numpy


def make_dataset(eq="vdp", p_true=S.MU_TRUE, sigma=S.SIGMA, n=S.N_POINTS, run=0, start=0, noise="gaussian"):
    """Returns (t, clean, noisy): times (n,), exact states (n, 2), noisy states (n, 2)."""
    y0 = S.START_STATES[start] if eq == "vdp" else (2.0, 0.0)
    t = np.linspace(0.0, S.T_END, n)
    sol = solve_ivp(rate_numpy(eq, p_true), (0.0, S.T_END), list(y0), t_eval=t, method="DOP853", rtol=1e-10, atol=1e-12)
    clean = sol.y.T

    # one seed per (system, true value, noise level, points, start state, run number)
    key = [S.BASE_SEED + (0 if eq == "vdp" else 1), round(p_true * 1000), round(sigma * 10000), n, start, run]
    rng = np.random.default_rng(np.random.SeedSequence(key))
    if noise == "laplace":                       # heavier tails, same standard deviation
        noisy = clean + rng.laplace(0.0, sigma / np.sqrt(2.0), size=clean.shape)
    else:
        noisy = clean + rng.normal(0.0, sigma, size=clean.shape)
    return t, clean, noisy
