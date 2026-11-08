"""Map step 2: the equations.

Van der Pol, second order:   x'' - mu (1 - x^2) x' + x = 0
Order reduction (v = x'):     x' = v,   v' = mu (1 - x^2) v - x
The state is (x, v); the rate of change is (x', v').

Duffing (second test system of the brief):  x' = v,  v' = -0.1 v - x - beta x^3, unknown beta (true 0.3).
"""
import torch

SYSTEMS = {
    #             true value, start guess, start state
    "vdp":     dict(p_true=1.5, p_guess=1.0, name="mu"),
    "duffing": dict(p_true=0.3, p_guess=0.2, name="beta"),
}


def rate(eq, x, v, p):
    """The rate (x', v') at state (x, v) for parameter p. Works for floats, numpy arrays and torch tensors."""
    if eq == "vdp":
        return v, p * (1 - x ** 2) * v - x
    return v, -0.1 * v - x - p * x ** 3


def rate_numpy(eq, p):
    """The same rule in the form SciPy's solver wants: f(t, y) -> [x', v']."""
    return lambda t, y: list(rate(eq, y[0], y[1], p))


def rate_torch(eq, y, p):
    """The same rule for a batch of states y[..., 2] in PyTorch (used in the physics loss)."""
    dx, dv = rate(eq, y[..., 0], y[..., 1], p)
    return torch.stack([dx, dv], -1)
