"""Every fixed number of the project in one place, with the reason it has this value."""

# ---- data (map step 3) ------------------------------------------------------------------------
T_END = 10.0                     # time span [0, 10]                        Kumar's setting
N_POINTS = 100                   # samples on [0, 10]                       Kumar's setting
SIGMA = 0.1                      # noise standard deviation                 Kumar's setting
MU_TRUE = 1.5                    # true Van der Pol parameter               Kumar's setting
START_STATES = [(2.0, 0.0),      # start state 0 = Kumar's (2, 0)
                (0.5, 0.5),      # extra start states, used only in the robustness experiment
                (-1.0, 1.5)]
BASE_SEED = 20261102             # all random numbers are derived from this + the settings of a run

# ---- unknown parameter ------------------------------------------------------------------------
MU_GUESS = 1.0                   # every method starts from mu = 1.0        Kumar's starting guess

# ---- networks (map steps 5-7) -----------------------------------------------------------------
INPUT_SCALE = 2.5                # states are divided by this before entering the network (values ~ -1..1)
INPUT_SCALE_LARGE_MU = 6.5       # at mu = 4 the velocity reaches ~6.3, so a wider scale keeps it on the KAN grid
MLP_WIDTH = 32                   # 2 -> 32 -> 32 -> 2, tanh (Kumar: 3 weight layers + tanh; width not given, our choice)
KAN_LAYERS = [2, 10, 2]          # default of the KAN-ODEs code
KAN_GRID = 5                     # 5 grid pieces on [-1, 1]
KAN_ORDER = 3                    # cubic splines  -> 5 + 3 = 8 spline bumps per wire

# ---- ODE solver (map step 8) ------------------------------------------------------------------
SOLVER = "dopri5"                # Dormand-Prince 5(4), adaptive step size  Kumar
RTOL = ATOL = 1e-3               # solver error tolerance                   Kumar

# ---- training (map steps 9-11) ----------------------------------------------------------------
PHYSICS_WEIGHT = 1.0             # loss = data + continuity + 1 x physics   Raissi (equal weights)
ADAM_STEPS = 1500                # Adam, learning rate 0.01                 Kumar uses Adam; budget fixed on test seeds
ADAM_LR = 1e-2
LBFGS_ITERS = 300                # then L-BFGS polish                       Raissi's recipe (Adam, then L-BFGS)
TRACE_EVERY = 100                # record losses and mu every 100 Adam steps

# ---- judging a result (map step 12) -----------------------------------------------------------
SUCCESS_MAX_ERROR = 0.10         # success: mu within 10% ...
FIT_FACTOR = 2.0                 # ... and data error <= 2 sigma^2 (the model fits down to the noise level)
