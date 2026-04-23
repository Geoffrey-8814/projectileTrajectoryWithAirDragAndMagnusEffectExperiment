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
if "roll_angle" not in st.session_state:
    st.session_state.roll_angle = 0.0
if "video_path" not in st.session_state:
    st.session_state.video_path = None
if "video_name" not in st.session_state:
    st.session_state.video_name = None
if "video_fps" not in st.session_state:
    st.session_state.video_fps = 30.0
if "video_total_frames" not in st.session_state:
    st.session_state.video_total_frames = 0

st.title("🎯 Projectile Motion Annotator")

# ── 2. SIDEBAR / INPUTS ─────────────────────────────────────────────
with st.sidebar:
    st.header("Settings")
    uploaded_video = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])
    wall_dist = st.number_input("Wall distance (m)", value=6.0, step=0.1)
    
    auto_advance = st.checkbox("Auto-advance frame after click", value=True)

    st.markdown("---")
    st.subheader("Camera Roll")

    def _on_roll_num():
        val = round(st.session_state._roll_num, 1)
        st.session_state.roll_angle = val
        st.session_state._roll_coarse = int(round(val))

    def _on_roll_coarse():
        val = float(st.session_state._roll_coarse)
        st.session_state.roll_angle = val
        st.session_state._roll_num = val

    st.number_input(
        "Roll angle (°)",
        min_value=-45.0, max_value=45.0,
        value=st.session_state.roll_angle, step=0.1, format="%.1f",
        key="_roll_num", on_change=_on_roll_num,
        help="Type or use arrows for 0.1° precision."
    )
    st.slider(
        "Coarse adjust",
        min_value=-45, max_value=45,
        value=int(round(st.session_state.roll_angle)),
        step=1, key="_roll_coarse", on_change=_on_roll_coarse,
        help="Drag for quick 1° steps."
    )

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


def rotate_frame(frame, angle_deg):
    """Rotate frame counter-clockwise by angle_deg degrees around its centre."""
    if angle_deg == 0:
        return frame
    h, w = frame.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(frame, M, (w, h))


def rotate_point(u, v, angle_deg, w, h):
    """Rotate pixel (u, v) counter-clockwise by angle_deg around frame centre.
    Uses the same Y-down convention as cv2.getRotationMatrix2D."""
    if angle_deg == 0:
        return u, v
    cx, cy = w / 2.0, h / 2.0
    theta = np.radians(angle_deg)
    dx, dy = u - cx, v - cy
    u_rot = cx + dx * np.cos(theta) + dy * np.sin(theta)
    v_rot = cy - dx * np.sin(theta) + dy * np.cos(theta)
    return int(round(u_rot)), int(round(v_rot))


def unrotate_point(u_rot, v_rot, angle_deg, w, h):
    """Inverse of rotate_point: recover original pixel from rotated-display pixel."""
    return rotate_point(u_rot, v_rot, -angle_deg, w, h)


def draw_reference_lines(frame, roll_deg):
    """Overlay a dashed crosshair and a spirit-level bubble gauge."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    overlay = frame.copy()

    # Dashed horizontal centre line
    dash, gap = 20, 8
    x = 0
    while x < w:
        cv2.line(overlay, (x, cy), (min(x + dash, w - 1), cy), (0, 220, 220), 1)
        x += dash + gap

    # Dashed vertical centre line
    y = 0
    while y < h:
        cv2.line(overlay, (cx, y), (cx, min(y + dash, h - 1)), (0, 220, 220), 1)
        y += dash + gap

    # Spirit-level gauge (bottom centre)
    bar_y = h - 28
    bar_half = 90
    cv2.line(overlay, (cx - bar_half, bar_y), (cx + bar_half, bar_y), (160, 160, 160), 2)
    # Target tick at centre (0°)
    cv2.line(overlay, (cx, bar_y - 9), (cx, bar_y + 9), (255, 255, 255), 2)
    # Bubble position: moves opposite to applied correction
    pixels_per_deg = bar_half / 15.0   # ±15° fills the bar
    bx = int(cx - roll_deg * pixels_per_deg)
    bx = max(cx - bar_half, min(cx + bar_half, bx))
    if abs(roll_deg) < 0.2:
        bcol = (0, 255, 0)        # green  – level
    elif abs(roll_deg) < 3.0:
        bcol = (0, 165, 255)      # orange – small tilt
    else:
        bcol = (0, 0, 255)        # red    – large tilt
    cv2.circle(overlay, (bx, bar_y), 8, bcol, -1)
    cv2.circle(overlay, (bx, bar_y), 8, (255, 255, 255), 1)   # white border
    # Label
    label = f"{roll_deg:+.1f}°"
    cv2.putText(overlay, label, (cx + bar_half + 6, bar_y + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    return frame


def get_world_coords(u, v, roll_deg=0.0):
    if camera_matrix is None: return 0, 0
    pts = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=None)
    x_n, y_n = undistorted[0, 0]
    # The camera is physically rolled by roll_deg. undistortPoints returns coords in
    # the camera's own tilted frame (Y-down). Rotating CCW by roll_deg in Y-down
    # convention realigns them with world axes (true horizontal/vertical).
    if roll_deg != 0:
        theta = np.radians(roll_deg)
        x_n, y_n = ( x_n * np.cos(theta) + y_n * np.sin(theta),
                    -x_n * np.sin(theta) + y_n * np.cos(theta))
    return float(x_n * wall_dist), float(y_n * wall_dist)


def read_frame(frame_idx):
    """Open a fresh VideoCapture, seek, read one frame, then immediately release.
    Avoids sharing a single cap across users/threads (libavcodec assertion crash).
    Retries with a sequential seek fallback for compressed formats where random
    seek to a non-keyframe can silently fail."""
    path = st.session_state.video_path
    if not path:
        return False, None

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return False, None

    # First attempt: direct random seek
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()

    # Fallback: sequential read from a slightly earlier position
    if not ret and frame_idx > 0:
        start = max(0, frame_idx - 5)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        for _ in range(frame_idx - start + 1):
            ret, frame = cap.read()
            if not ret:
                break

    cap.release()
    return ret, frame

# ── 4. MAIN ANNOTATOR ───────────────────────────────────────────────
if uploaded_video and camera_matrix is not None:
    # Write the video to a per-session temp file only when a new file is uploaded
    if st.session_state.video_name != uploaded_video.name:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded_video.read())
            st.session_state.video_path = tmp.name
        st.session_state.video_name = uploaded_video.name
        cap_meta = cv2.VideoCapture(st.session_state.video_path)
        st.session_state.video_fps = cap_meta.get(cv2.CAP_PROP_FPS)
        st.session_state.video_total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
        cap_meta.release()

    fps = st.session_state.video_fps
    total_frames = st.session_state.video_total_frames

    # --- Process any pending canvas click BEFORE rendering (eliminates double-rerun) ---
    # streamlit_image_coordinates stores its last value in st.session_state[key], so we
    # can read and act on a click at the TOP of the script, update frame_idx here, and
    # render the correct frame below — all within the single natural rerun the click
    # component already triggered. No st.rerun() needed.
    _ck = f"canvas_{st.session_state.frame_idx}_{st.session_state.roll_angle}"
    _raw = st.session_state.get(_ck)
    if _raw is not None and (_raw["x"], _raw["y"]) != st.session_state.last_processed_click:
        _u, _v = unrotate_point(int(_raw["x"]), int(_raw["y"]),
                                st.session_state.roll_angle, DISPLAY_W, DISPLAY_H)
        _xw, _yw = get_world_coords(_u, _v, roll_deg=st.session_state.roll_angle)
        st.session_state.annotations[st.session_state.frame_idx] = {
            "timestamp": st.session_state.frame_idx / fps,
            "u": _u, "v": _v, "x": _xw, "y": _yw
        }
        st.session_state.last_processed_click = (_raw["x"], _raw["y"])
        if auto_advance and st.session_state.frame_idx < total_frames - 1:
            st.session_state.frame_idx += 1

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
    ret, frame = read_frame(st.session_state.frame_idx)
    if not ret:
        st.warning(f"⚠️ Could not decode frame {st.session_state.frame_idx}. "
                   "Try stepping to the previous or next frame.")
    if ret:
        frame = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))

        # Apply camera roll rotation to the display frame
        display_frame = rotate_frame(frame, st.session_state.roll_angle)

        # Draw reference crosshair + level gauge
        display_frame = draw_reference_lines(display_frame, st.session_state.roll_angle)

        # Draw all existing annotations (rotate stored original coords to match display)
        for f, data in st.session_state.annotations.items():
            color = (0, 0, 255) if f == st.session_state.frame_idx else (0, 255, 0)
            du, dv = rotate_point(data["u"], data["v"],
                                   st.session_state.roll_angle, DISPLAY_W, DISPLAY_H)
            cv2.circle(display_frame, (du, dv), 5, color, -1)

        # UI Overlay
        roll_label = f" | Roll: {st.session_state.roll_angle:+.1f}°" if st.session_state.roll_angle != 0 else ""
        cv2.putText(display_frame,
                    f"Frame: {st.session_state.frame_idx} | Time: {st.session_state.frame_idx/fps:.2f}s{roll_label}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # --- Coordinate Input ---
        img_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        streamlit_image_coordinates(
            Image.fromarray(img_rgb),
            key=f"canvas_{st.session_state.frame_idx}_{st.session_state.roll_angle}"
        )

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