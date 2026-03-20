import streamlit as st

st.set_page_config(layout="wide", page_title="User Guide")

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

st.title("📖 How to use the Projectile Analysis Tool")

# Sidebar for quick navigation
st.sidebar.header("Contents")
st.sidebar.markdown("""
- [1. Recording Video](#1-camera-setup-video-capture-critical)
- [2. Calibration](#2-camera-calibration)
- [3. Annotation](#3-annotation-the-logger)
- [4. Optimization](#4-trajectory-optimization)
- [5. Distillation](#5-robot-distillation)
""")

# Load the raw tutorial text
try:
    with open("tutorial.txt", "r") as f:
        tutorial_text = f.read()
except FileNotFoundError:
    tutorial_text = "Tutorial file not found. Please ensure tutorial.txt exists."

# ─── SECTION 1: VIDEO CAPTURE ───
st.header("🎥 1. Camera Setup & Video Capture")
col1, col2 = st.columns([2, 1])
with col1:
    st.info("""
    **The Golden Rule:** The camera must be **perpendicular** to the plane of motion. 
    If the projectile moves closer to or further from the camera during flight, the 2D-to-3D 
    math will fail, and your velocities will be incorrect.
    """)
    st.markdown("""
    *   **Use a Tripod:** Handheld video is unusable for coordinate tracking.
    *   **Lock Exposure/Focus:** Prevent the camera from "hunting" for focus while the ball is moving.
    *   **Measure Distance:** Use a laser measure or tape to find the distance from the camera lens to the ball's flight path.
    """)
with col2:
    # A placeholder for a diagram or icon
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Angle_of_view.svg/300px-Angle_of_view.svg.png", caption="Maintain a 90-degree side-view")

st.divider()

# ─── SECTION 2: CALIBRATION ───
st.header("🏁 2. Camera Calibration")
st.write("""
Most cameras have "lens distortion." To get accurate meters-per-second, we must remove this.
""")
c1, c2 = st.columns(2)
with c1:
    st.subheader("The Board")
    st.write("""
    1. Generate the ChArUco board in the **Calibration Page**.
    2. Print it (ensure 'Fit to Page' is OFF; it must be actual size).
    3. Measure the squares with a ruler to confirm they match your settings.
    """)
with c2:
    st.subheader("The Video")
    st.write("""
    1. Record a video of the board.
    2. **Crucial:** Move the board so it covers all 4 corners and the center of the screen.
    3. Ensure the board is **filling the frame** in many shots to give the algorithm enough data to detect the lens curve.
    """)

st.divider()

# ─── SECTION 3: WORKFLOW ───
st.header("📈 3. The Analysis Workflow")

tab1, tab2, tab3 = st.tabs(["Annotation", "Optimization", "Distillation"])

with tab1:
    st.subheader("Step 1: Coordinate Logging")
    st.write("""
    Upload your video and click the center of the projectile in every frame. 
    The app uses your Calibration file and the 'Wall Distance' to turn pixels into real meters.
    """)
    st.success("Tip: Use 'Auto-advance' to make the app jump to the next frame automatically after a click.")

with tab2:
    st.subheader("Step 2: Physics Fitting")
    st.write("""
    The experimental data (your clicks) is compared against a physics model. 
    The optimizer adjusts $V_0$, $\\theta$, $C_d$, and $C_l$ until the model matches your video.
    This gives you the most accurate launch parameters possible.
    """)

with tab3:
    st.subheader("Step 3: Robot Export")
    st.write("""
    Computers are fast, but Robot Controllers (like RoboRIO) cannot run complex 
    physics simulations 50 times a second. The **Distiller** creates a simple 
    "Look-up Formula" (Polynomial) that mimics the physics engine but runs instantly.
    """)

# ─── DOWNLOAD BUTTON ───
st.divider()
st.subheader("Offline Copy")
st.download_button("📥 Download Tutorial as TXT", tutorial_text, "tutorial.txt")