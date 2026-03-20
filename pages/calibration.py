import io
import tempfile
import cv2
import numpy as np
import streamlit as st
from cv2.aruco import CharucoBoard, CharucoDetector, DICT_4X4_100


# ── 1. Board Generation ────────────────────────────────────────────

st.header("1. Generate ChArUco Board")

col1, col2 = st.columns(2)
with col1:
    squares_x = st.number_input("Columns (squaresX)", min_value=3, max_value=30, value=18)
    squares_y = st.number_input("Rows (squaresY)", min_value=3, max_value=30, value=11)
with col2:
    square_length = st.number_input("Square side (m)", value=0.01258571, format="%.8f")
    marker_length = st.number_input("Marker side (m)", value=0.009229523, format="%.8f")

img_w = st.number_input("Image width (px)", value=2360, step=10)
img_h = st.number_input("Image height (px)", value=1640, step=10)

dictionary = cv2.aruco.getPredefinedDictionary(DICT_4X4_100)

if st.button("Generate board image"):
    board = CharucoBoard((int(squares_x), int(squares_y)), square_length, marker_length, dictionary)
    board_image = board.generateImage((int(img_w), int(img_h)))
    st.image(board_image, caption="ChArUco Board", use_container_width=True)

    _, png_buf = cv2.imencode(".png", board_image)
    st.download_button(
        "Download board PNG",
        data=png_buf.tobytes(),
        file_name="charuco_board.png",
        mime="image/png",
    )
    st.session_state["board_params"] = {
        "squares_x": int(squares_x),
        "squares_y": int(squares_y),
        "square_length": square_length,
        "marker_length": marker_length,
    }

st.markdown("---")

# ── 2. Calibration ─────────────────────────────────────────────────

st.header("2. Calibrate from Video")

st.info(
    "Upload a calibration video where the ChArUco board is visible from many angles. "
    "The board parameters above will be used for detection."
)

uploaded_video = st.file_uploader("Upload calibration video", type=["mp4", "avi", "mov", "mkv"])
resize_w = st.number_input("Resize width (px, 0 = original)", min_value=0, value=1280, step=10)
resize_h = st.number_input("Resize height (px, 0 = original)", min_value=0, value=720, step=10)
min_corners = st.slider("Min detected corners per frame", 4, 20, 6)
max_frames = st.slider("Max frames to use", 15, 500, 100)
frame_step = st.slider("Process every Nth frame", 1, 30, 1)

if uploaded_video is not None and st.button("Run calibration", type="primary"):
    # Board from current sidebar values (or session state)
    bp = st.session_state.get("board_params", {
        "squares_x": int(squares_x),
        "squares_y": int(squares_y),
        "square_length": square_length,
        "marker_length": marker_length,
    })
    board = CharucoBoard(
        (bp["squares_x"], bp["squares_y"]),
        bp["square_length"],
        bp["marker_length"],
        dictionary,
    )
    detector = CharucoDetector(board)

    # Write uploaded bytes to a temp file so OpenCV can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_video.read())
        tmp_path = tmp.name

    cap = cv2.VideoCapture(tmp_path)
    if not cap.isOpened():
        st.error("Could not open the uploaded video.")
        st.stop()

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_object_points = []
    all_image_points = []
    image_size = None
    kept = 0

    progress = st.progress(0, text="Detecting corners...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % frame_step != 0:
            continue

        if resize_w > 0 and resize_h > 0:
            frame = cv2.resize(frame, (resize_w, resize_h))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)

        if charuco_ids is not None and len(charuco_ids) >= min_corners:
            object_points, image_points = board.matchImagePoints(charuco_corners, charuco_ids)
            all_object_points.append(object_points)
            all_image_points.append(image_points)
            image_size = gray.shape[::-1]
            kept += 1

        progress.progress(
            min(frame_idx / total_frames, 1.0),
            text=f"Frame {frame_idx}/{total_frames} — kept {kept}",
        )

        if kept >= max_frames:
            break

    cap.release()
    progress.empty()

    if kept < 15:
        st.error(f"Only {kept} usable frames found (need at least 15). Try a different video or lower the min-corners threshold.")
        st.stop()

    st.info(f"Running calibration on {kept} frames...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points,
        all_image_points,
        image_size,
        None,
        None,
    )

    # Store in session state
    st.session_state["calibration"] = {
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rms_error": ret,
        "image_size": image_size,
        "frames_used": kept,
    }

    st.success(f"Calibration complete — RMS reprojection error: **{ret:.4f}** px  ({kept} frames used)")

# ── 3. Results & Download ──────────────────────────────────────────

calib = st.session_state.get("calibration")
if calib is not None:
    st.markdown("---")
    st.header("3. Calibration Results")

    st.markdown(f"**RMS error:** {calib['rms_error']:.4f} px &nbsp;|&nbsp; **Frames used:** {calib['frames_used']} &nbsp;|&nbsp; **Image size:** {calib['image_size']}")

    st.subheader("Camera Matrix")
    st.dataframe(calib["camera_matrix"], use_container_width=True)

    st.subheader("Distortion Coefficients")
    st.write(calib["dist_coeffs"].ravel().tolist())

    # Download as .npz
    buf = io.BytesIO()
    np.savez(
        buf,
        camera_matrix=calib["camera_matrix"],
        dist_coeffs=calib["dist_coeffs"],
        resolution=calib["image_size"],
    )
    buf.seek(0)
    st.download_button(
        "Download calibration.npz",
        data=buf,
        file_name="calibration.npz",
        mime="application/octet-stream",
    )
