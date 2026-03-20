import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from ballistics import solve_projectile, extract_apex
from project_params import load_project_params

# --- SETTINGS ---
PROJECT_NAME = "fuel_2026"
PROJECT = load_project_params(PROJECT_NAME)
APEX_BEFORE_TARGET = PROJECT["apex_before_target"]
G = PROJECT["gravity"]

def simulate(vx, vy):
    sol = solve_projectile(vx, vy, PROJECT, t_span=(0, 4), hit_ground=False, max_step=0.05)
    y_max, x_apex = extract_apex(sol)
    
    return sol.y[0], sol.y[1], y_max, x_apex

last_solution = [None]

def solve_for_params(target_x, target_y, target_hmax):
    def obj(p):
        tx, ty, h, x_apex = simulate(p[0], p[1])
        y_at_x = np.interp(target_x, tx, ty)
        
        error = (y_at_x - target_y)**2 + (h - target_hmax)**2 + (max(0, target_x - tx[-1])**2)
        
        if APEX_BEFORE_TARGET:
            if x_apex > target_x: error += (x_apex - target_x)**2 * 10
        else:
            if x_apex < target_x: error += (target_x - x_apex)**2 * 10
            
        return error

    default_start = [target_x * 1.2, 12] if APEX_BEFORE_TARGET else [target_x * 2.5, 8]
    starts = [default_start]
    if last_solution[0] is not None:
        starts.insert(0, last_solution[0])

    best_res = None
    for s in starts:
        res = minimize(obj, s, method='Nelder-Mead',
                       options={'maxiter': 10000, 'xatol': 1e-9, 'fatol': 1e-9})
        if best_res is None or res.fun < best_res.fun:
            best_res = res
        if best_res.fun < 1e-6:
            break
    
    last_solution[0] = best_res.x.tolist()
    return best_res.x, best_res.fun

# --- 2. VACUUM BASELINE (standard kinematics) ---
def vacuum_velocities(dx, dy, hmax):
    """Analytically compute (vx, vy) for a vacuum trajectory hitting (dx, dy) with apex at hmax.
    
    From kinematics:
      vy = sqrt(2 * g * hmax)            (apex height equation)
      dy = vy*(dx/vx) - g*(dx/vx)^2/2   (trajectory at target)
    Rearranging the second equation into a quadratic in vx:
      dy*vx^2 - vy*dx*vx + g*dx^2/2 = 0
    """
    vy = np.sqrt(2 * G * hmax)
    
    if abs(dy) < 1e-10:
        # Special case: target at same height as launch
        vx = G * dx / (2 * vy)
    else:
        disc = vy**2 - 2 * G * dy  # = 2*g*(hmax - dy), always >= 0 when hmax > dy
        if disc < 0:
            return None, None
        sqrt_disc = np.sqrt(disc)
        if APEX_BEFORE_TARGET:
            # Minus root → steeper angle → apex before target
            vx = dx * (vy - sqrt_disc) / (2 * dy)
        else:
            # Plus root → flatter angle → apex after target
            vx = dx * (vy + sqrt_disc) / (2 * dy)
    
    return vx, vy

if __name__ == "__main__":
    # --- 3. GENERATE DATASET (store deltas: actual - vacuum) ---
    dy_const = PROJECT["default_target_dy"]

    CONVERGENCE_THRESH = 1e-4
    print(f"Generating data for '{PROJECT_NAME}' (Apex before target: {APEX_BEFORE_TARGET})...")
    X_data, Y_data = [], []
    skipped = 0
    max_cost = 0.0

    for dx in np.linspace(0.5, 8, 20):
        last_solution[0] = None
        for hmax in np.linspace(dy_const + 1, dy_const + 3.0, 10):
            # Vacuum baseline
            vx_vac, vy_vac = vacuum_velocities(dx, dy_const, hmax)
            if vx_vac is None or vx_vac <= 0:
                skipped += 1
                continue
            
            # Physics-optimized (with drag + Magnus)
            (vx, vy), cost = solve_for_params(dx, dy_const, hmax)
            if cost > CONVERGENCE_THRESH:
                skipped += 1
                continue
            
            max_cost = max(max_cost, cost)
            X_data.append([dx, hmax])
            Y_data.append([vx - vx_vac, vy - vy_vac])  # DELTA only

    print(f"Kept {len(X_data)} points, skipped {skipped} (cost > {CONVERGENCE_THRESH})")
    print(f"Worst kept cost: {max_cost:.2e}")

    if not X_data:
        raise RuntimeError(
            "No valid training samples were generated. "
            "Adjust dy/hmax ranges, convergence threshold, or project parameters in params/projects.json."
        )

    X_data, Y_data = np.array(X_data), np.array(Y_data)

    # --- 4. REGRESSION on aero correction ---
    poly = PolynomialFeatures(degree=3)
    X_poly = poly.fit_transform(X_data)
    model = LinearRegression()
    model.fit(X_poly, Y_data)

    Y_pred = model.predict(X_poly)
    print(f"\nR² Score — ΔVx: {r2_score(Y_data[:,0], Y_pred[:,0]):.6f}, ΔVy: {r2_score(Y_data[:,1], Y_pred[:,1]):.6f}")

    # --- 5. EXPORT FOR ROBORIO ---
    print("\n--- COPY THESE TO ROBORIO ---")
    full_dvx = model.coef_[0].copy(); full_dvx[0] = model.intercept_[0]
    full_dvy = model.coef_[1].copy(); full_dvy[0] = model.intercept_[1]

    feat_names = poly.get_feature_names_out(['dx', 'hmax'])
    print(f"// Deployment steps:")
    print(f"//   1. vy_vac = sqrt(2 * {G} * hmax)")
    print(f"//   2. vx_vac = dx * (vy_vac - sqrt(vy_vac^2 - 2*{G}*{dy_const:.4f})) / (2*{dy_const:.4f})")
    print(f"//   3. dvx = dot(coeffs_dvx, features),  dvy = dot(coeffs_dvy, features)")
    print(f"//   4. vx = vx_vac + dvx,  vy = vy_vac + dvy")
    print(f"// Features: {feat_names.tolist()}")
    print(f"dVx Coeffs: {full_dvx.tolist()}")
    print(f"dVy Coeffs: {full_dvy.tolist()}")

    # --- 6. SELF-TEST ---
    test_pts = [[2.0, 2.5], [4.5, 3.0], [7.0, 4.0]]
    print("\n--- SELF-TEST ---")
    for pt in test_pts:
        dx, hmax = pt
        vx_vac, vy_vac = vacuum_velocities(dx, dy_const, hmax)
        if vx_vac is None:
            print(f"  dx={dx}, hmax={hmax}: vacuum impossible")
            continue
        
        pred_delta = model.predict(poly.transform([pt]))[0]
        vx_final = vx_vac + pred_delta[0]
        vy_final = vy_vac + pred_delta[1]
        
        (vx_exact, vy_exact), cost = solve_for_params(dx, dy_const, hmax)
        
        print(f"  dx={dx}, hmax={hmax}:")
        print(f"    vacuum  = ({vx_vac:.4f}, {vy_vac:.4f})")
        print(f"    exact   = ({vx_exact:.4f}, {vy_exact:.4f})")
        print(f"    model   = ({vx_final:.4f}, {vy_final:.4f})")
        print(f"    delta   = ({pred_delta[0]:.4f}, {pred_delta[1]:.4f})")
        print(f"    error   = ({vx_final-vx_exact:.6f}, {vy_final-vy_exact:.6f})")

    # --- 7. VISUALIZATION ---
    plt.figure(figsize=(14, 10))

    # Plot 1: Trajectory comparison
    plt.subplot(2, 2, 1)
    test_dx, test_dy, test_h = 3.6, dy_const, 3.0
    (vx_p, vy_p), _ = solve_for_params(test_dx, test_dy, test_h)
    tx_p, ty_p, _, _ = simulate(vx_p, vy_p)

    vx_v, vy_v = vacuum_velocities(test_dx, test_dy, test_h)
    pred_d = model.predict(poly.transform([[test_dx, test_h]]))[0]
    tx_m, ty_m, _, _ = simulate(vx_v + pred_d[0], vy_v + pred_d[1])

    plt.plot(tx_p, ty_p, 'r-', label='Exact Physics', linewidth=3, alpha=0.5)
    plt.plot(tx_m, ty_m, 'b--', label='Vacuum + Aero Correction')
    plt.scatter([test_dx], [test_dy], color='red', zorder=5, label='Target')
    plt.title(f"Trajectory @ {test_dx}m (Apex Before: {APEX_BEFORE_TARGET})")
    plt.legend(); plt.grid(True)

    # Plot 2: Aero correction magnitudes
    plt.subplot(2, 2, 2)
    plt.scatter(X_data[:, 0], Y_data[:, 0], s=10, label='ΔVx (aero correction)')
    plt.scatter(X_data[:, 0], Y_data[:, 1], s=10, label='ΔVy (aero correction)')
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    plt.xlabel("Distance (m)"); plt.ylabel("ΔVelocity (m/s)")
    plt.title("Aero Correction Magnitude")
    plt.legend(); plt.grid(True)

    # Plot 3: Polynomial fit residuals
    plt.subplot(2, 2, 3)
    residuals = Y_data - Y_pred
    plt.scatter(X_data[:, 0], residuals[:, 0], s=10, label='ΔVx residual')
    plt.scatter(X_data[:, 0], residuals[:, 1], s=10, label='ΔVy residual')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel("Distance (m)"); plt.ylabel("Residual (m/s)")
    plt.title("Polynomial Fit Residuals")
    plt.legend(); plt.grid(True)

    # Plot 4: Multiple trajectories
    plt.subplot(2, 2, 4)
    for dx_test in [2, 4, 6]:
        hmax_test = dy_const + 2.0
        (vx_e, vy_e), _ = solve_for_params(dx_test, dy_const, hmax_test)
        tx_e, ty_e, _, _ = simulate(vx_e, vy_e)
        
        vv, vyv = vacuum_velocities(dx_test, dy_const, hmax_test)
        pd = model.predict(poly.transform([[dx_test, hmax_test]]))[0]
        tx_m, ty_m, _, _ = simulate(vv + pd[0], vyv + pd[1])
        
        plt.plot(tx_e, ty_e, '-', linewidth=2, alpha=0.5, label=f'{dx_test}m exact')
        plt.plot(tx_m, ty_m, '--', label=f'{dx_test}m model')
        plt.scatter([dx_test], [dy_const], color='red', zorder=5)

    plt.title("Multiple Distance Comparison")
    plt.legend(fontsize=8); plt.grid(True)

    plt.tight_layout()
    plt.savefig('distill_result.png', dpi=100)
    print("Saved plot to distill_result.png")
    plt.show()