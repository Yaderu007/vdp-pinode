"""Step 0: check that the environment is complete and the official repositories are the ones being used."""
import sys

import torch
import torchdiffeq
import efficient_kan
import numpy, scipy, pandas, matplotlib  # noqa: F401

from vdp.networks import count_weights, make_network

print("python      ", sys.version.split()[0], "at", sys.executable)
print("torch       ", torch.__version__, "| GPU visible:", torch.cuda.is_available(), "(the experiments use the CPU)")
print("torchdiffeq  loaded from", list(torchdiffeq.__path__)[0])
print("efficient_kan loaded from", list(efficient_kan.__path__)[0])
for k in ("mlp", "mlp16", "kan"):
    print(f"{k:6s} weights:", count_weights(make_network(k)))
print("all good" if "repos" in list(torchdiffeq.__path__)[0] and "repos" in list(efficient_kan.__path__)[0]
      else "WARNING: torchdiffeq or efficient_kan is not loaded from your cloned repos (see README, step 3)")
