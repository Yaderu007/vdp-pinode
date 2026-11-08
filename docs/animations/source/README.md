# Animation clips

`animation.html` holds all 8 scenes (open it in a browser to watch them play). Each scene is one entry in the
`SCENES` list inside the file, with its own `draw(t)` function and duration `dur`: edit one entry to change one clip.
The data (exact solution, the 100 noisy points, training snapshots of run 0, trained MLP weights) is embedded as
`const D = {...}`; `make_data_snapshots.py` is the script that produced it.

| clip | scene |
|---|---|
| 1_system | state moving along the exact Van der Pol path |
| 2_sampling | 100 samples appear, then noise |
| 3_mlp | states flow through the MLP, out comes the rate arrow |
| 4_kan | the same with the KAN (curves on wires, schematic) |
| 5_arrow_map | the network's rate at every state; network vs physics at one state |
| 6_ode_solver | three solver steps, then the full trajectory |
| 7_training | windows, arrows, µ and the losses over training |
| 8_result | the trained model's trajectory over the data |

Re-render: `./make_videos.sh` (all) or `./make_videos.sh 3 6` (only clips 3 and 6) -> `videos/`.
