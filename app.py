import os
# Disable FFmpeg async frame-threading to prevent libavcodec assertion failures
# (pthread_frame.c:175) on Linux cloud environments (Streamlit Cloud).
os.environ["OPENCV_FFMPEG_MULTITHREADED"] = "0"

import streamlit as st

st.set_page_config(
    page_title="Projectile Trajectory Lab",
    layout="wide",
)

pg = st.navigation([
    st.Page("views/tutorial.py", title="User Guide", icon="📖"),
    st.Page("views/configuration.py", title="Project Configuration", icon="🛠"),
    st.Page("views/calibration.py", title="Camera Calibration", icon="📷"),
    st.Page("views/annotation.py", title="Video Annotation", icon="🎯"),
    st.Page("views/trajectory.py", title="Trajectory Analysis", icon="🚀"),
    st.Page("views/distill.py", title="Controller Distiller", icon="🤖"),
    
])
pg.run()

