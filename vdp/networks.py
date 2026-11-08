"""Map steps 5-7: input scaling, the MLP and the KAN.

Both networks do the same job: 2 numbers in (the scaled state), 2 numbers out (the rate x', v').
"""
import torch
from efficient_kan import KAN          # the official efficient-kan repository (Blealtan), installed from your clone

from . import settings as S


class MLP(torch.nn.Module):
    """2 -> width -> width -> 2 with tanh. Each layer: output = tanh(weights @ input + biases); no tanh at the end.
    Weights: (2*32 + 32) + (32*32 + 32) + (32*2 + 2) = 96 + 1056 + 66 = 1218 for width 32."""

    def __init__(self, width=S.MLP_WIDTH):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(2, width), torch.nn.Tanh(),
            torch.nn.Linear(width, width), torch.nn.Tanh(),
            torch.nn.Linear(width, 2))
        for layer in self.layers:                     # Xavier start values, zero biases (as in Raissi's code)
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(self, u):
        return self.layers(u)


def make_network(kind):
    """kind: 'mlp' (1218 weights), 'mlp16' (a small MLP with 354 weights) or 'kan' (400 weights).
    KAN: 40 wires x (8 spline heights + 1 base weight + 1 scale) = 400."""
    if kind == "mlp":
        return MLP(S.MLP_WIDTH)
    if kind == "mlp16":
        return MLP(16)
    if kind == "kan":
        return KAN(S.KAN_LAYERS, grid_size=S.KAN_GRID, spline_order=S.KAN_ORDER)
    raise ValueError(f"unknown network '{kind}'")


def count_weights(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class RateField(torch.nn.Module):
    """The learned rule: state (x, v) -> rate (x', v'). Divides the state by the input scale, then calls the network.
    torchdiffeq calls this as field(t, y); our system does not depend on t, so t is ignored."""

    def __init__(self, net, scale=S.INPUT_SCALE):
        super().__init__()
        self.net, self.scale = net, scale

    def forward(self, t, y):
        shape = y.shape
        return self.net((y / self.scale).reshape(-1, 2)).reshape(shape)
