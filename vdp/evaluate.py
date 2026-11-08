"""Map step 12: judging a result.

relative error = |learned - true| / true
success        = error <= 10%  and  data error <= 2 sigma^2 (the model fits the dots down to the noise level)
failure types  = crash | no_fit (cannot match the dots) | wrong_parameter (fits, but parameter off by > 10%)
Discovery R^2  = how well the learned rule matches the true rule on a 100 x 100 grid of states (Spotorno et al.)
"""
import numpy as np
import torch

from . import settings as S
from .system import rate


def judge(result, p_true, sigma):
    if result["status"] != "ok":
        return dict(rel_error=np.nan, success=False, failure="crash")
    err = abs(result["p_hat"] - p_true) / p_true
    fits = result["data_mse"] <= S.FIT_FACTOR * sigma ** 2
    failure = "none" if (fits and err <= S.SUCCESS_MAX_ERROR) else ("no_fit" if not fits else "wrong_parameter")
    return dict(rel_error=err, success=failure == "none", failure=failure)


def discovery_r2(result, eq, p_true, box=2.5, states=None):
    """R^2 between the learned and the true unknown part of the equation (vdp: mu (1 - x^2) v).
    On the grid x, v in [-box, box] (brief), or only at the given states (e.g. the clean trajectory)."""
    if result["status"] != "ok":
        return np.nan
    if states is None:
        g = np.linspace(-box, box, 100)
        X, V = np.meshgrid(g, g, indexing="ij")
        states = np.stack([X.ravel(), V.ravel()], -1)
    x, v = states[:, 0], states[:, 1]
    known = -x if eq == "vdp" else -0.1 * v - x            # the part of v' that does not contain the parameter
    true = rate(eq, x, v, p_true)[1] - known
    if result["method"] == "ls":                            # least squares learns the formula itself
        learned = rate(eq, x, v, result["p_hat"])[1] - known
    else:
        with torch.no_grad():
            learned = result["field"](None, torch.tensor(states)).numpy()[:, 1] - known
    return float(1 - ((learned - true) ** 2).sum() / ((true - true.mean()) ** 2).sum())
