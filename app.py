import streamlit as st

st.set_page_config(
    page_title="Projectile Trajectory Lab",
    layout="wide",
)

pg = st.navigation([
    st.Page("pages/tutorial.py", title="User Guide", icon="📖"),
    st.Page("pages/Configuration.py", title="Project Configuration", icon="🛠"),
    st.Page("pages/calibration.py", title="Camera Calibration", icon="📷"),
    st.Page("pages/annotation.py", title="Video Annotation", icon="🎯"),
    st.Page("pages/trajectory.py", title="Trajectory Analysis", icon="🚀"),
    st.Page("pages/distill.py", title="Controller Distiller", icon="🤖"),
    
    
])
pg.run()
