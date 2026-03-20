import streamlit as st
import json
import pandas as pd

st.set_page_config(layout="wide", page_title="Project Parameter Generator")

st.title("🛠 Project Parameter Generator")
st.markdown("""
Use this page to define physical parameters for different projectile types. 
Once you have added your projects, download the JSON file.
""")

# --- 1. SESSION STATE INITIALIZATION ---
if "project_list" not in st.session_state:
    # Starting with your provided examples as defaults
    st.session_state.project_list = {
        "fuel_2026": {
            "mass": 0.27, "radius": 0.076, "gravity": 9.8, "air_density": 1.225,
            "cd": 0.212, "cl": 1.0, "v_delta_ratio": 0.75, 
            "apex_before_target": True, "default_target_dy": 1.2288
        }
    }

# --- 2. ADD NEW PROJECT FORM ---
with st.expander("➕ Add New Project Profile", expanded=True):
    with st.form("new_project_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            name = st.text_input("Project Unique ID", placeholder="e.g., basketball_v1")
            mass = st.number_input("Mass (kg)", value=0.1, format="%.4f")
            radius = st.number_input("Radius (m)", value=0.05, format="%.4f")
            
        with col2:
            gravity = st.number_input("Gravity (m/s²)", value=9.81, format="%.2f")
            air_density = st.number_input("Air Density (kg/m³)", value=1.225, format="%.3f")
            target_dy = st.number_input("Default Target Δy (m)", value=0.0, format="%.4f")

        with col3:
            cd = st.number_input("Drag Coeff (Cd)", value=0.47, format="%.3f")
            cl = st.number_input("Lift Coeff (Cl)", value=0.0, format="%.3f")
            v_ratio = st.number_input("V Delta Ratio", value=0.0, format="%.2f")
            apex = st.checkbox("Apex before target", value=True)

        submit = st.form_submit_button("Add Project to List")
        
        if submit:
            if name and name not in st.session_state.project_list:
                st.session_state.project_list[name] = {
                    "mass": mass, "radius": radius, "gravity": gravity,
                    "air_density": air_density, "cd": cd, "cl": cl,
                    "v_delta_ratio": v_ratio, "apex_before_target": apex,
                    "default_target_dy": target_dy
                }
                st.success(f"Added '{name}'")
            elif name in st.session_state.project_list:
                st.error("This ID already exists. Delete it below first to overwrite.")
            else:
                st.error("Please provide a Unique ID.")

# --- 3. MANAGE & VIEW PROJECTS ---
st.divider()
st.subheader("Current Projects")

if st.session_state.project_list:
    # Convert to DataFrame for easy viewing
    df = pd.DataFrame.from_dict(st.session_state.project_list, orient="index")
    st.dataframe(df, use_container_width=True)

    # Delete Functionality
    to_delete = st.selectbox("Select a project to remove", options=list(st.session_state.project_list.keys()))
    if st.button(f"🗑 Delete {to_delete}"):
        del st.session_state.project_list[to_delete]
        st.rerun()

    # --- 4. EXPORT ---
    st.divider()
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("JSON Preview")
        final_json = {"projects": st.session_state.project_list}
        st.json(final_json)

    with col_right:
        st.subheader("Export")
        json_string = json.dumps(final_json, indent=2)
        st.download_button(
            label="📥 Download projects.json",
            data=json_string,
            file_name="projects.json",
            mime="application/json"
        )
else:
    st.info("No projects added yet.")