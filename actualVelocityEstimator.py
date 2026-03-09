import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares
import pandas as pd
import os

# ── Physical Constants ──────────────────────────────────────────────
# 2026 FRC "FUEL" Game Piece Specifications (from traj.py)
m = 0.27           # Mass (kg)
r = 0.12           # Ball radius (m)
A = np.pi * r**2    # Projected area (m²)
g = 9.81            # Gravity (m/s²)
rho = 1.225         # Air density (kg/m³)

# Default aerodynamic coefficients (can also be optimized)
Cd_default = 0.47   # Drag coefficient for foam sphere
Cl_default = 0.15   # Lift coefficient (Magnus effect)

Cd_default = 0.2   # Drag coefficient for foam sphere
Cl_default = 0.3   # Lift coefficient (Magnus effect)




def read_coordinates(path):
    """Read and normalize trajectory data from CSV."""
    data = pd.read_csv(path)
    x = data["x"].values
    y = data["y"].values
    t = data["timestamp"].values

    # Swap axes and flip y (matching trajOptimizer convention)
    # x, y = y.copy(), x.copy()
    y *= -1
    x *= -1

    # Normalize to origin
    x -= x[0]
    y -= y[0]
    t -= t[0]

    return t, x, y


def simulate_trajectory(v0, theta, Cd, Cl, t_eval):
    """
    Simulate projectile motion with drag + Magnus effect.
    Returns the ODE solution object.
    """
    omega = 0.5 * v0 / (2*r)  # Angular velocity (backspin estimate)

    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    initial_state = [0.0, 0.0, vx0, vy0]

    def deriv(t, state):
        x, y, vx, vy = state
        v = np.hypot(vx, vy)

        # Drag force
        Fd = 0.5 * rho * Cd * A * v**2
        Fd_x = -Fd * vx / v if v > 1e-8 else 0.0
        Fd_y = -Fd * vy / v if v > 1e-8 else 0.0

        # Magnus force (2D cross-product simplification)
        FM_coeff = 0.5 * rho * A * r * Cl
        FM_x = FM_coeff * (-omega * vy)
        FM_y = FM_coeff * (omega * vx)

        ax = (Fd_x + FM_x) / m
        ay = (Fd_y + FM_y) / m - g
        return [vx, vy, ax, ay]

    sol = solve_ivp(
        deriv,
        t_span=[0, max(t_eval) + 1],
        y0=initial_state,
        t_eval=t_eval,
        method='RK45',
        atol=1e-9,
        rtol=1e-6,
    )
    return sol


def build_objective(t_data, x_data, y_data, optimize_aero=False):
    """
    Build a residual function for least_squares.

    Early data points are weighted more heavily (exponential decay with
    normalized time) because the trajectory becomes noisier / more
    unpredictable the longer the ball is in the air.

    If optimize_aero=True, params = [v0, theta, Cd, Cl]
    Otherwise,               params = [v0, theta]
    """
    # ── Time-based weights: early points get ~3× weight of last point ──
    t_norm = t_data / t_data[-1]          # 0 → 1
    weights = np.exp(-1.0 * t_norm)       # e^0 ≈ 1.0  →  e^-1 ≈ 0.37
    weights /= weights.mean()             # normalise so mean weight ≈ 1

    def objective(params):
        if optimize_aero:
            v0, theta, Cd, Cl = params
        else:
            v0, theta = params
            Cd, Cl = Cd_default, Cl_default

        sol = simulate_trajectory(v0, theta, Cd, Cl, t_data)

        if len(sol.t) == 0:
            return np.full(2 * len(x_data), 1e6)

        x_sim = np.interp(t_data, sol.t, sol.y[0])
        y_sim = np.interp(t_data, sol.t, sol.y[1])

        # Apply weights so early-trajectory residuals dominate the cost
        return np.concatenate([weights * (x_sim - x_data),
                               weights * (y_sim - y_data)])

    return objective


def estimate(path, optimize_aero=False, plot=True):
    """
    Estimate launch velocity and angle for a given trial.

    Parameters
    ----------
    speed : int
        Nominal launcher speed setting (e.g. 40, 50, 60).
    index : int
        Trial index (1, 2, 3).
    optimize_aero : bool
        If True, also optimize Cd and Cl alongside v0 and theta.
    plot : bool
        Show a comparison plot.

    Returns
    -------
    dict with optimized v0, theta (deg), and optionally Cd, Cl.
    """
    t_data, x_data, y_data = read_coordinates(path)

    # ── Initial guesses ─────────────────────────────────────────────
    # Rough v0 from displacement / time
    dx = x_data[1] - x_data[0]
    dy_max = y_data[1] - y_data[0]
    dt = t_data[1] - t_data[0]
    v0_guess = np.hypot(dx, 2 * dy_max) / dt
    theta_guess = np.arctan2(2 * dy_max, abs(dx))  # crude angle estimate
    
    # v0_guess = 7.5
    # theta_guess = 60 / (2*3.14)  # crude angle estimate

    if optimize_aero:
        params0 = [v0_guess, theta_guess, Cd_default, Cl_default]
        lower = [1.0,  -np.pi / 2, 0.1, 0.0]
        upper = [50.0,  np.pi / 2, 2.0, 1.0]
    else:
        params0 = [v0_guess, theta_guess]
        lower = [1.0,  -np.pi / 2]
        upper = [50.0,  np.pi / 2]

    objective = build_objective(t_data, x_data, y_data, optimize_aero)

    result = least_squares(
        objective,
        params0,
        bounds=(lower, upper),
        loss='soft_l1',
        f_scale=0.1,
        verbose=0,
    )

    # ── Extract results ─────────────────────────────────────────────
    if optimize_aero:
        v0_opt, theta_opt, Cd_opt, Cl_opt = result.x
    else:
        v0_opt, theta_opt = result.x
        Cd_opt, Cl_opt = Cd_default, Cl_default

    theta_deg = np.degrees(theta_opt)

    print(f"\n{'='*50}")
    print(f"{'='*50}")
    print(f"  Initial Velocity (v0) : {v0_opt:.2f} m/s")
    print(f"  Launch Angle (θ)      : {theta_deg:.2f}°")
    if optimize_aero:
        print(f"  Drag Coeff (Cd)       : {Cd_opt:.4f}")
        print(f"  Lift Coeff (Cl)       : {Cl_opt:.4f}")
    print(f"  Residual cost         : {result.cost:.6f}")
    print(f"{'='*50}")

    # ── Plot ────────────────────────────────────────────────────────
    if plot:
        t_fine = np.linspace(0, t_data[-1], 300)
        sol = simulate_trajectory(v0_opt, theta_opt, Cd_opt, Cl_opt, t_fine)

        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data, 'ro', markersize=6, label='Experimental Data')
        plt.plot(sol.y[0], sol.y[1], 'b-', linewidth=2,
                 label=f'Fit: v0={v0_opt:.1f} m/s, θ={theta_deg:.1f}°')
        plt.title(f'Launch Estimation — {path}', fontsize=14)
        plt.xlabel('Horizontal Distance (m)')
        plt.ylabel('Vertical Height (m)')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return {
        'v0': v0_opt,
        'theta_deg': theta_deg,
        'Cd': Cd_opt,
        'Cl': Cl_opt,
        'cost': result.cost,
    }


# ── Main: run a single trial ───────────────────────────────────────
if __name__ == '__main__':
    # path = r"dataset\testingShooter\3.6-1-annotations.csv"
    path = r"dataset\fuels\15-30-2-annotations.csv"
    
    
    # path = r"dataset\fuels\20-30-2-annotations.csv"
    res = estimate(path, optimize_aero=False, plot=True)
    print(res)
