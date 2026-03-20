import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import time
from scipy.optimize import minimize
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- IMPORT YOUR CUSTOM MODULES ---
from ballistics import solve_projectile, extract_apex

# st.set_page_config(layout="wide", page_title="Robot Distiller")

st.title("🤖 Robot Controller Distiller")
st.markdown("""
This tool generates polynomial coefficients for your robot. 
It fits a model to the **delta** between simple vacuum math and complex aero physics.
""")

# ── 1. SIDEBAR CONFIGURATION ───────────────────────────────────────

with st.sidebar:
    st.header("1. Load Project")
    project_file = st.file_uploader("Upload projects.json", type=["json"])
    
    selected_project = None
    if project_file:
        config = json.load(project_file)
        selected_project_name = st.selectbox("Select Project Profile", list(config["projects"].keys()))
        params = config["projects"][selected_project_name]
        st.success(f"Loaded: {selected_project_name}")
    else:
        st.warning("Please upload projects.json")
        st.stop()

    st.header("2. Search Ranges")
    col_dx1, col_dx2 = st.columns(2)
    dist_min = col_dx1.number_input("Min Distance (m)", value=1.0)
    dist_max = col_dx2.number_input("Max Distance (m)", value=8.0)
    
    target_dy = st.number_input("Target Height Δy (m)", value=params.get("default_target_dy", 1.22))
    
    col_h1, col_h2 = st.columns(2)
    h_offset_min = col_h1.number_input("Min Apex Offset (m)", value=1.0, help="Height above target")
    h_offset_max = col_h2.number_input("Max Apex Offset (m)", value=3.0)

    st.header("3. Model Settings")
    poly_degree = 3
    num_samples = st.slider("Samples (Density)", 5, 30, 15)

# ── 2. CORE LOGIC ──────────────────────────────────────────────────

def vacuum_velocities(dx, dy, hmax, gravity, apex_before):
    """Analytically compute (vx, vy) for vacuum."""
    vy = np.sqrt(2 * gravity * hmax)
    if abs(dy) < 1e-10:
        vx = gravity * dx / (2 * vy)
    else:
        disc = vy**2 - 2 * gravity * dy
        if disc < 0: return None, None
        sqrt_disc = np.sqrt(disc)
        # Apex before target = steeper = lower vx for same height
        vx = dx * (vy - sqrt_disc) / (2 * dy) if apex_before else dx * (vy + sqrt_disc) / (2 * dy)
    return vx, vy

def solve_for_params(target_x, target_y, target_hmax, project_params):
    """Finds exact vx, vy using physics engine."""
    apex_before = project_params["apex_before_target"]
    
    def obj(p):
        sol = solve_projectile(p[0], p[1], project_params, t_span=(0, 4), hit_ground=False)
        if len(sol.t) == 0: return 1e6
        y_at_x = np.interp(target_x, sol.y[0], sol.y[1])
        h_max, x_apex = extract_apex(sol)
        
        error = (y_at_x - target_y)**2 + (h_max - target_hmax)**2
        # Penalty for wrong apex side
        if apex_before and x_apex > target_x: error += (x_apex - target_x)**2 * 10
        if not apex_before and x_apex < target_x: error += (target_x - x_apex)**2 * 10
        return error

    # Initial guess based on vacuum
    v0_vac = vacuum_velocities(target_x, target_y, target_hmax, project_params["gravity"], apex_before)
    start_p = v0_vac if v0_vac[0] is not None else [target_x, 10]
    
    res = minimize(obj, start_p, method='Nelder-Mead', options={'xatol': 1e-7, 'fatol': 1e-7})
    return res.x, res.fun

# ── 3. EXECUTION ───────────────────────────────────────────────────

if st.button("🚀 Run Distillation", type="primary", use_container_width=True):
    X_data, Y_data = [], []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    dx_range = np.linspace(dist_min, dist_max, num_samples)
    hmax_range = np.linspace(target_dy + h_offset_min, target_dy + h_offset_max, 8)
    total_steps = len(dx_range) * len(hmax_range)
    step = 0

    start_time = time.time()
    
    for dx in dx_range:
        for hmax in hmax_range:
            vx_vac, vy_vac = vacuum_velocities(dx, target_dy, hmax, params["gravity"], params["apex_before_target"])
            
            if vx_vac is not None and vx_vac > 0:
                (vx_phys, vy_phys), cost = solve_for_params(dx, target_dy, hmax, params)
                
                if cost < 1e-3:
                    X_data.append([dx, hmax])
                    Y_data.append([vx_phys - vx_vac, vy_phys - vy_vac])
            
            step += 1
            progress_bar.progress(step / total_steps)
            status_text.text(f"Processing: Distance {dx:.2f}m, Apex {hmax:.2f}m")

    duration = time.time() - start_time
    st.success(f"Distillation complete in {duration:.1f} seconds. Generated {len(X_data)} valid data points.")

    if len(X_data) > 0:
        X_data, Y_data = np.array(X_data), np.array(Y_data)

        # Regression
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(X_data)
        model = LinearRegression()
        model.fit(X_poly, Y_data)
        
        Y_pred = model.predict(X_poly)
        
        # Coefficients
        coeffs_vx = model.coef_[0].copy(); coeffs_vx[0] = model.intercept_[0]
        coeffs_vy = model.coef_[1].copy(); coeffs_vy[0] = model.intercept_[1]
        feat_names = poly.get_feature_names_out(['dx', 'hmax'])

        # --- Display Results ---
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric("R² Score (Vx Delta)", f"{r2_score(Y_data[:,0], Y_pred[:,0]):.6f}")
        with col_res2:
            st.metric("R² Score (Vy Delta)", f"{r2_score(Y_data[:,1], Y_pred[:,1]):.6f}")

        # --- Plots ---
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot 1: Delta Magnitudes
        axes[0].scatter(X_data[:, 0], Y_data[:, 0], s=15, label='ΔVx (Aero Correction)', alpha=0.6)
        axes[0].scatter(X_data[:, 0], Y_data[:, 1], s=15, label='ΔVy (Aero Correction)', alpha=0.6)
        axes[0].set_title("Aero Correction Required")
        axes[0].set_xlabel("Distance (m)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot 2: Fit Residuals
        axes[1].scatter(X_data[:, 0], Y_data[:, 0] - Y_pred[:, 0], s=10, label='Vx Residual')
        axes[1].scatter(X_data[:, 0], Y_data[:, 1] - Y_pred[:, 1], s=10, label='Vy Residual')
        axes[1].axhline(0, color='black', lw=1)
        axes[1].set_title("Model Error (Residuals)")
        axes[1].set_xlabel("Distance (m)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        st.pyplot(fig)

        # --- Export / Code Snippet ---
        st.divider()
        st.subheader("📋 Robot Implementation Data")
        
        output_data = {
            "project": selected_project_name,
            "degree": poly_degree,
            "target_dy": target_dy,
            "features": feat_names.tolist(),
            "coeffs_dvx": coeffs_vx.tolist(),
            "coeffs_dvy": coeffs_vy.tolist()
        }

        col_code, col_json = st.columns(2)
        
        with col_code:
            st.markdown("**Java / C++ Pseudo-code**")
            code_str = f"""// 1. Calculate vacuum base
double vy_vac = Math.sqrt(2 * {params['gravity']} * hmax);
double disc = Math.pow(vy_vac, 2) - 2 * {params['gravity']} * {target_dy:.4f};
double vx_vac = dx * (vy_vac - Math.sqrt(disc)) / (2 * {target_dy:.4f});

// 2. Apply Polynomial Correction
double dvx = 0; double dvy = 0;
double[] features = {{
            1, dx, hmax,
            dx2, dx * hmax, h2,
            dx3, dx2 * hmax, dx * h2, h3
        }};
double[] coeffsVx = {json.dumps(coeffs_vx.tolist())};
double[] coeffsVy = {json.dumps(coeffs_vy.tolist())};

for(int i=0; i<coeffsVx.length; i++) {{
    dvx += features[i] * coeffsVx[i];
    dvy += features[i] * coeffsVy[i];
}}
double finalVx = vx_vac + dvx;
double finalVy = vy_vac + dvy;"""
            st.code(code_str, language='java')

        with col_json:
            st.markdown("**JSON Output**")
            st.json(output_data)
            st.download_button(
                "📥 Download Robot Config",
                data=json.dumps(output_data, indent=2),
                file_name=f"robot_coeffs_{selected_project_name}.json",
                mime="application/json"
            )

else:
    st.info("Click 'Run Distillation' to generate robot coefficients. This may take a minute.")