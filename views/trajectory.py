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

st.set_page_config(layout="wide", page_title="Trajectory Optimizer")

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
    loss_func = st.selectbox("Loss Function", ["linear", "soft_l1", "huber"], index=1)
    
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

    return t_proc, x_proc, y_proc

# ── 3. CORE PHYSICS & OPTIMIZATION LOGIC ─────────────────────────────

def simulate_trajectory(v0, theta, Cd, Cl, t_eval, project_params):
    """Wraps your solve_projectile_from_polar."""
    # Update local params with optimized values
    p = dict(project_params)
    p["cd"] = Cd
    p["cl"] = Cl

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

def run_optimization(t_data, x_data, y_data, project_params, opt_aero):
    # Initial Guesses (derived from your code)
    dx = x_data[1] - x_data[0]
    dy = y_data[1] - y_data[0]
    dt = t_data[1] - t_data[0]
    v0_guess = np.hypot(dx, 2 * dy) / dt
    theta_guess = np.arctan2(2 * dy, abs(dx))

    # Weighting: early points get more weight (from your code)
    t_norm = t_data / t_data[-1]
    weights = np.exp(-1.0 * t_norm)
    weights /= weights.mean()

    def objective(vars):
        if opt_aero:
            v0, theta, Cd, Cl = vars
        else:
            v0, theta = vars
            Cd, Cl = project_params["cd"], project_params["cl"]

        sol = simulate_trajectory(v0, theta, Cd, Cl, t_data, project_params)
        
        if len(sol.t) == 0:
            return np.full(2 * len(x_data), 1e6)

        x_sim = np.interp(t_data, sol.t, sol.y[0])
        y_sim = np.interp(t_data, sol.t, sol.y[1])

        return np.concatenate([weights * (x_sim - x_data),
                               weights * (y_sim - y_data)])

    # Bounds
    if opt_aero:
        params0 = [v0_guess, theta_guess, project_params["cd"], project_params["cl"]]
        lower = [1.0,  -np.pi/2, 0.1, 0.0]
        upper = [50.0,  np.pi/2, 2.0, 1.5]
    else:
        params0 = [v0_guess, theta_guess]
        lower = [1.0,  -np.pi/2]
        upper = [50.0,  np.pi/2]

    res = least_squares(objective, params0, bounds=(lower, upper), loss=loss_func, f_scale=0.1)
    return res

# ── 4. MAIN UI ───────────────────────────────────────────────────────

uploaded_csv = st.file_uploader("Upload Annotations CSV", type=["csv"])

if uploaded_csv and params:
    df_raw = pd.read_csv(uploaded_csv)
    t_data, x_data, y_data = preprocess_data(df_raw)
    
    col_run, col_info = st.columns([1, 3])
    with col_run:
        run_btn = st.button("▶ Run Optimization", type="primary", use_container_width=True)
    
    if run_btn:
        with st.spinner("Optimizing trajectory..."):
            result = run_optimization(t_data, x_data, y_data, params, optimize_aero)
            
            # Extract Results
            if optimize_aero:
                v0_opt, theta_opt, Cd_opt, Cl_opt = result.x
            else:
                v0_opt, theta_opt = result.x
                Cd_opt, Cl_opt = params["cd"], params["cl"]
            
            # --- Metrics Display ---
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Launch Velocity", f"{v0_opt:.2f} m/s")
            m2.metric("Launch Angle", f"{np.degrees(theta_opt):.2f}°")
            m3.metric("Optimized Cd", f"{Cd_opt:.4f}")
            m4.metric("Residual Cost", f"{result.cost:.6f}")

            # --- Plotting ---
            t_fine = np.linspace(0, t_data[-1], 200)
            sol = simulate_trajectory(v0_opt, theta_opt, Cd_opt, Cl_opt, t_fine, params)

            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(x_data, y_data, 'ro', label='Experimental Data (Normalized)', alpha=0.6)
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
                "Parameter": ["v0 (m/s)", "Angle (deg)", "Cd", "Cl", "Cost"],
                "Value": [v0_opt, np.degrees(theta_opt), Cd_opt, Cl_opt, result.cost]
            })
            st.table(res_df)

elif not params:
    st.warning("👈 Please upload and select a Project Profile in the sidebar first.")
else:
    st.info("Please upload an annotations CSV file to begin.")