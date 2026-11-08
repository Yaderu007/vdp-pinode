"""Physics-informed Neural ODEs (MLP and KAN) vs. classical least squares for the Van der Pol oscillator.

One file per step of the project map:
    settings.py      every fixed number, with the reason for it
    system.py        step 2  - the equations (Van der Pol, Duffing)
    data.py          step 3  - making the noisy datasets
    networks.py      steps 5-7 - input scaling, the MLP and the KAN
    neural_ode.py    steps 4, 8-11 - solver, shooting windows, loss, training
    least_squares.py step 15 - the classical baseline (no network)
    evaluate.py      step 12 - error, success rule, Discovery R^2
"""
