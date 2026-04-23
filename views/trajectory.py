import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import json
import io

# --- IMPORT YOUR CUSTOM MODULES ---
# Ensure these files are in the same directory as your streamlit app
from ballistics import solve_projectile_from_polar 


st.title("🚀 Projectile Trajectory Optimizer")
st.markdown("Estimate launch velocity and angle by fitting experimental data to a physics model.")

# ── 1. LOAD CONFIGURATIONS ──────────────────────────────────────────

with st.sidebar:
    st.header("1. Setup Parameters")
    
    # Load Projects
    project_file = st.file_uploader("Upload projects.json", type=["json"])
    selected_project = None
    params = None
    
    if project_file:
        config = json.load(project_file)
        project_names = list(config["projects"].keys())
        selected_project_name = st.selectbox("Select Project Profile", project_names)
        params = config["projects"][selected_project_name]
        st.success(f"Loaded: {selected_project_name}")
    else:
        st.info("Please upload the projects.json generated in the previous page.")

    st.divider()
    st.header("2. Optimization Settings")
    optimize_aero = st.checkbox("Optimize Aero (Cd & Cl)", value=False)
    optimize_spin = st.checkbox("Optimize spin rate (v_delta_ratio)", value=False,
                                help="Also fit the spin rate. Recommended when aero is on "
                                     "to avoid Cl/spin degeneracy.")
    loss_func = st.selectbox("Loss Function", ["linear", "soft_l1", "huber"], index=0)
    
# ── 2. DATA LOADING & PRE-PROCESSING ────────────────────────────────

def preprocess_data(df):
    """Matches your read_coordinates logic."""
    # Create a copy to avoid modifying session state
    data = df.copy()
    
    x = data["x"].values
    y = data["y"].values
    t = data["timestamp"].values

    # Your specific coordinate transformations
    if x[0] > x[1]:
        x_proc = x * -1
    else:
        x_proc = x.copy()
    y_proc = y * -1
    

    # Normalize to origin (starting point 0,0,0)
    x_proc -= x_proc[0]
    y_proc -= y_proc[0]
    t_proc = t - t[0]

    # Auto-detect millisecond timestamps: if total flight > 100, it's almost
    # certainly not in seconds (a 100-second flight is impossible here).
    if t_proc[-1] > 100.0:
        t_proc = t_proc / 1000.0

    return t_proc, x_proc, y_proc

# ── 3. CORE PHYSICS & OPTIMIZATION LOGIC ─────────────────────────────

def simulate_trajectory(v0, theta, Cd, Cl, t_eval, project_params, vdr=None):
    """Wraps your solve_projectile_from_polar."""
    # Update local params with optimized values
    p = dict(project_params)
    p["cd"] = Cd
    p["cl"] = Cl
    if vdr is not None:
        p["v_delta_ratio"] = vdr

    sol = solve_projectile_from_polar(
        v0=v0,
        theta=theta,
        params=p,
        t_span=[0, max(t_eval) + 1],
        t_eval=t_eval,
        hit_ground=False,
        max_step=0.05,
    )
    return sol

def run_optimization(t_data, x_data, y_data, project_params, opt_aero, opt_spin):
    g = project_params["gravity"]

    # --- Guess 1: median of first few inter-frame velocities ---
    n_init = min(4, len(t_data) - 1)
    vx_ests = [(x_data[i+1] - x_data[i]) / max(t_data[i+1] - t_data[i], 1e-9)
               for i in range(n_init)]
    vy_ests = [(y_data[i+1] - y_data[i]) / max(t_data[i+1] - t_data[i], 1e-9)
               for i in range(n_init)]
    vx_g1 = float(np.median(vx_ests))
    vy_g1 = float(np.median(vy_ests))

    # --- Guess 2: from apex geometry ---
    # At apex vy=0, so vy0 ≈ g*t_apex; vx0 ≈ x_apex/t_apex
    apex_idx = int(np.argmax(y_data))
    t_apex = float(t_data[apex_idx])
    x_apex = float(x_data[apex_idx])
    vy_g2 = g * t_apex
    vx_g2 = x_apex / max(t_apex, 1e-9)

    def _clamp(v0, theta):
        return (float(np.clip(v0, 1.5, 49.0)),
                float(np.clip(theta, -np.pi/2 + 0.01, np.pi/2 - 0.01)))

    # Primary guesses (reported / used for plot) — fix: use vy_g2, not vx_g2 twice
    v0_guess, theta_guess = _clamp(np.hypot(vx_g2, vy_g2),
                                   np.arctan2(vy_g2, abs(vx_g2)))

    candidate_starts = [
        _clamp(np.hypot(vx_g1, vy_g1), np.arctan2(vy_g1, abs(vx_g1))),
        _clamp(np.hypot(vx_g2, vy_g2), np.arctan2(vy_g2, abs(vx_g2))),
    ]
    for v0_s in np.linspace(4.0, 30.0, 5):
        for deg_s in np.linspace(10.0, 65.0, 5):
            candidate_starts.append(_clamp(v0_s, np.radians(deg_s)))

    # Normalize residuals so x and y contribute equally regardless of trajectory aspect ratio
    x_scale = max(np.max(np.abs(x_data)), 1e-3)
    y_scale = max(np.max(np.abs(y_data)), 1e-3)

    def _build_params(vars):
        if opt_aero and opt_spin:
            v0, theta, Cd, Cl, vdr = vars
        elif opt_aero:
            v0, theta, Cd, Cl = vars
            vdr = None
        else:
            v0, theta = vars
            Cd, Cl = project_params["cd"], project_params["cl"]
            vdr = None
        return v0, theta, Cd, Cl, vdr

    def objective(vars):
        v0, theta, Cd, Cl, vdr = _build_params(vars)
        sol = simulate_trajectory(v0, theta, Cd, Cl, t_data, project_params, vdr=vdr)
        if len(sol.t) == 0:
            return np.full(2 * len(x_data), 1e6)
        x_sim = np.interp(t_data, sol.t, sol.y[0])
        y_sim = np.interp(t_data, sol.t, sol.y[1])
        return np.concatenate([(x_sim - x_data) / x_scale,
                               (y_sim - y_data) / y_scale])

    Cd0 = project_params["cd"]
    Cl0 = project_params["cl"]
    vdr0 = project_params["v_delta_ratio"]

    if opt_aero and opt_spin:
        lower = [1.0, -np.pi/2, 0.05, 0.0, 0.0]
        upper = [50.0,  np.pi/2, 1.5,  4.0, 1]  # Cl up to 3.0; ratio < 2 keeps v_slow >= 0
        def make_p0(v0, th): return [v0, th, Cd0, Cl0, vdr0]
    elif opt_aero:
        lower = [1.0, -np.pi/2, 0.05, 0.0]
        upper = [50.0,  np.pi/2, 1.5,  4.0]  # Cl up to 3.0
        def make_p0(v0, th): return [v0, th, Cd0, Cl0]
    else:
        lower = [1.0,  -np.pi/2]
        upper = [50.0,  np.pi/2]
        def make_p0(v0, th): return [v0, th]

    # ── Phase 1: fast scan — find best basin ────────────────────────
    best_cost = np.inf
    best_p0 = make_p0(*candidate_starts[1])
    for v0_s, th_s in candidate_starts:
        try:
            r = least_squares(objective, make_p0(v0_s, th_s),
                              bounds=(lower, upper), loss=loss_func,
                              f_scale=1.0, method='trf', max_nfev=80)
            if r.cost < best_cost:
                best_cost = r.cost
                best_p0 = r.x.tolist()
        except Exception:
            pass

    # ── Phase 2: full refinement from the best basin ─────────────────
    res = least_squares(objective, best_p0, bounds=(lower, upper),
                        loss=loss_func, f_scale=1.0,
                        method='trf', max_nfev=5000,
                        ftol=1e-10, xtol=1e-10, gtol=1e-10)
    return res, v0_guess, theta_guess

# ── 4. MAIN UI ───────────────────────────────────────────────────────

uploaded_csv = st.file_uploader("Upload Annotations CSV", type=["csv"])

if uploaded_csv and params:
    df_raw = pd.read_csv(uploaded_csv)
    t_data, x_data, y_data = preprocess_data(df_raw)

    dt_raw = df_raw["timestamp"].iloc[1] - df_raw["timestamp"].iloc[0]
    ts_unit = "ms (auto-converted to s)" if dt_raw > 0.5 else "s"
    st.caption(f"📋 {len(t_data)} points · Δt={t_data[1]-t_data[0]:.4f} s · "
               f"duration={t_data[-1]:.3f} s · timestamp unit detected: {ts_unit}")
    
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("▶ Run Optimization", type="primary", use_container_width=True)
    
    if run_btn:
        with st.spinner("Optimizing trajectory..."):
            result, v0_guess, theta_guess = run_optimization(
                t_data, x_data, y_data, params, optimize_aero, optimize_spin)
            
            # Extract Results
            if optimize_aero and optimize_spin:
                v0_opt, theta_opt, Cd_opt, Cl_opt, vdr_opt = result.x
            elif optimize_aero:
                v0_opt, theta_opt, Cd_opt, Cl_opt = result.x
                vdr_opt = params["v_delta_ratio"]
            else:
                v0_opt, theta_opt = result.x
                Cd_opt, Cl_opt = params["cd"], params["cl"]
                vdr_opt = params["v_delta_ratio"]
            
            # --- Metrics Display ---
            st.divider()
            # RMSE in physical metres
            sol_at_data = simulate_trajectory(v0_opt, theta_opt, Cd_opt, Cl_opt, t_data, params,
                                              vdr=vdr_opt)
            if len(sol_at_data.t) > 0:
                x_fit = np.interp(t_data, sol_at_data.t, sol_at_data.y[0])
                y_fit = np.interp(t_data, sol_at_data.t, sol_at_data.y[1])
                rmse = float(np.sqrt(np.mean((x_fit-x_data)**2 + (y_fit-y_data)**2)))
            else:
                rmse = float('nan')

            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Launch Velocity", f"{v0_opt:.2f} m/s")
            m2.metric("Launch Angle", f"{np.degrees(theta_opt):.2f}°")
            m3.metric("Cd", f"{Cd_opt:.4f}")
            m4.metric("Cl", f"{Cl_opt:.4f}")
            m5.metric("v_delta_ratio", f"{vdr_opt:.3f}")
            m6.metric("RMSE", f"{rmse:.4f} m")

            # --- Plotting ---
            t_fine = np.linspace(0, t_data[-1], 200)
            sol_guess = simulate_trajectory(v0_guess, theta_guess,
                                            params["cd"], params["cl"],
                                            t_fine, params)
            sol = simulate_trajectory(v0_opt, theta_opt, Cd_opt, Cl_opt, t_fine, params,
                                      vdr=vdr_opt)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_data, y_data, 'ro', label='Experimental Data (Normalized)', alpha=0.6, zorder=3)
            if len(sol_guess.t) > 0:
                ax.plot(sol_guess.y[0], sol_guess.y[1], color='gray', linestyle='--',
                        linewidth=1.5, alpha=0.6, label=f'Initial Guess (v₀={v0_guess:.1f} m/s, θ={np.degrees(theta_guess):.1f}°)')
            ax.plot(sol.y[0], sol.y[1], 'b-', linewidth=2, label='Best Fit Model')
            
            ax.set_xlabel('Horizontal Distance (m)')
            ax.set_ylabel('Vertical Height (m)')
            ax.set_title(f'Trajectory Fit: {selected_project_name}')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()
            
            st.pyplot(fig)

            # --- Export Results ---
            st.subheader("Results Summary")
            res_df = pd.DataFrame({
                "Parameter": ["v0 (m/s)", "Angle (deg)", "Cd", "Cl", "v_delta_ratio", "RMSE (m)"],
                "Value": [v0_opt, np.degrees(theta_opt), Cd_opt, Cl_opt, vdr_opt, rmse]
            })
            st.table(res_df)

elif not params:
    st.warning("👈 Please upload and select a Project Profile in the sidebar first.")
else:
    st.info("Please upload an annotations CSV file to begin.")