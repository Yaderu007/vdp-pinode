import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# 1. Define the Van der Pol system of differential equations
def vanderpol_deriv(t, state, mu):
    x, y = state
    dxdt = y
    dydt = mu * (1 - x**2) * y - x
    return [dxdt, dydt]

# 2. Simulation parameters
mu = 1.5           # Damping parameter (adjust this to change nonlinear distortion)
t_span = (0, 100)   # Time span to simulate (seconds)
t_eval = np.linspace(t_span[0], t_span[1], 2000) # Smooth plotting points

# Initial conditions [x0, y0] -> starts slightly off-center to show the outward spiral
initial_state = [1, 1]

# 3. Solve the system numerically using an adaptive RK45 method
solution = solve_ivp(vanderpol_deriv, t_span, initial_state, args=(mu,), t_eval=t_eval)

x_vals = solution.y[0]
y_vals = solution.y[1]
time_steps = solution.t

# 4. Create an interactive double-plot figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot A: Phase Space (x vs dy/dt)
ax1.plot(x_vals, y_vals, label=f'Trajectory ($\mu$={mu})', color='royalblue', lw=1.5)
ax1.plot(initial_state[0], initial_state[1], 'ro', label='Start ($x_0$={initial_state[0]}, $y_0$={initial_state[1]})')
ax1.set_title('Phase Space Portrait')
ax1.set_xlabel('Position ($x$)')
ax1.set_ylabel('Velocity ($y = \dot{x}$)')
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend()

# Plot B: Time Series Behavior
ax2.plot(time_steps, x_vals, label='Position ($x$)', color='darkorange', lw=1.5)
ax2.plot(time_steps, y_vals, label='Velocity ($y$)', color='teal', lw=1.2, linestyle=':')
ax2.set_title('Time Series Evolution')
ax2.set_xlabel('Time ($t$)')
ax2.set_ylabel('Amplitude')
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend()

plt.tight_layout()
plt.show()