import tempfile
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image


# ── 1. SESSION STATE INITIALIZATION ────────────────────────────────
if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0
if "annotations" not in st.session_state:
    st.session_state.annotations = {}
if "last_processed_click" not in st.session_state:
    st.session_state.last_processed_click = None

st.title("🎯 Projectile Motion Annotator")

# ── 2. SIDEBAR / INPUTS ─────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    wall_dist = st.number_input("Wall distance (m)", value=6.0, step=0.1)
    
    auto_advance = st.checkbox("Auto-advance frame after click", value=True)
    
    st.markdown("---")
    calib_source = st.radio("Calibration", ["Session", "Upload .npz"])
    camera_matrix, dist_coeffs = None, None

    if calib_source == "Session":
        if "calibration" in st.session_state:
            camera_matrix = st.session_state.calibration["camera_matrix"]
            dist_coeffs = st.session_state.calibration["dist_coeffs"]
            resolution = st.session_state.calibration["image_size"]
            DISPLAY_W, DISPLAY_H = resolution
            st.success("Calibration loaded")
    else:
        npz = st.file_uploader("Upload .npz", type=["npz"])
        if npz:
            data = np.load(npz)
            camera_matrix, dist_coeffs, resolution = data["camera_matrix"], data["dist_coeffs"], data["resolution"]
            DISPLAY_W, DISPLAY_H = resolution

# ── 3. HELPERS ──────────────────────────────────────────────────────


def get_world_coords(u, v):
    if camera_matrix is None: return 0, 0
    pts = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=None)
    x_n, y_n = undistorted[0, 0]
    return float(x_n * wall_dist), float(y_n * wall_dist)

@st.cache_resource
def get_cap(video_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_file.read())
        return cv2.VideoCapture(tmp.name)

# ── 4. MAIN ANNOTATOR ───────────────────────────────────────────────
if uploaded_video and camera_matrix is not None:
    cap = get_cap(uploaded_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # --- Navigation Controls ---
    c1, c2, c3, c4 = st.columns([4, 1, 1, 2])
    
    with c1:
        # Use a callback to update the session state frame_idx when slider moves
        def on_slider_change():
            st.session_state.frame_idx = st.session_state.slider_val

        st.slider("Frame Navigation", 0, total_frames - 1, 
                  key="slider_val", 
                  value=st.session_state.frame_idx,
                  on_change=on_slider_change)

    with c2:
        if st.button("◀ Prev") and st.session_state.frame_idx > 0:
            st.session_state.frame_idx -= 1
            st.rerun()
    with c3:
        if st.button("Next ▶") and st.session_state.frame_idx < total_frames - 1:
            st.session_state.frame_idx += 1
            st.rerun()
    with c4:
        if st.button("🗑 Delete Point"):
            st.session_state.annotations.pop(st.session_state.frame_idx, None)
            st.rerun()

    # --- Load and Draw Frame ---
    cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        
        # Draw all existing annotations
        for f, data in st.session_state.annotations.items():
            color = (0, 0, 255) if f == st.session_state.frame_idx else (0, 255, 0)
            cv2.circle(frame, (data["u"], data["v"]), 5, color, -1)
        
        # UI Overlay
        cv2.putText(frame, f"Frame: {st.session_state.frame_idx} | Time: {st.session_state.frame_idx/fps:.2f}s", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- Coordinate Input ---
        # The key changes ONLY when the frame_idx changes. 
        # This fixes the "two-click" bug because the widget resets for the new frame.
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        coords = streamlit_image_coordinates(
            Image.fromarray(img_rgb),
            key=f"canvas_{st.session_state.frame_idx}" 
        )

        # --- Click Logic ---
        if coords is not None:
            curr_click = (coords["x"], coords["y"])
            
            # Check if this is a new click (prevents loop)
            if curr_click != st.session_state.last_processed_click:
                u, v = int(coords["x"]), int(coords["y"])
                x_world, y_world = get_world_coords(u, v)
                
                # Store data
                st.session_state.annotations[st.session_state.frame_idx] = {
                    "timestamp":st.session_state.frame_idx / fps, "u":u, "v":v, "x":x_world, "y":y_world
                }
                
                st.session_state.last_processed_click = curr_click
                
                # Logger Pro Style: Move to next frame automatically
                if auto_advance and st.session_state.frame_idx < total_frames - 1:
                    st.session_state.frame_idx += 1
                
                st.rerun()

    # ── 5. DATA TABLE ────────────────────────────────────────────────
    if st.session_state.annotations:
        st.divider()
        df = pd.DataFrame.from_dict(st.session_state.annotations, orient="index")
        df.index.name = "frame"
        df = df.sort_index().reset_index()
        
        col_tab, col_btn = st.columns([3, 1])
        with col_tab:
            st.dataframe(df, use_container_width=True, height=200)
        with col_btn:
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download CSV", csv, "data.csv", "text/csv")
            if st.button("🔥 Clear All Data"):
                st.session_state.annotations = {}
                st.session_state.frame_idx = 0
                st.rerun()

elif not uploaded_video:
    st.info("Upload a video to start.")
else:
    st.warning("Missing camera calibration data.")