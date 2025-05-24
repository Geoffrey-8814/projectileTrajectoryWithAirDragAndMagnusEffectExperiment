import cv2
import numpy as np
import pandas as pd
# Load the video
name = "50-3"
video_path = f'videos\\{name}.mp4'
cap = cv2.VideoCapture(video_path)

camera_matrix = np.array([[773.37970801,   0.,         958.76279554],
    [  0.,         580.10162599, 539.05443012],
    [  0.,           0.,           1.        ]], dtype=np.float64)

dist_coeffs = np.array([3.64748840e-04, -4.20081883e-03, -6.74638597e-05, -1.74431482e-04,
    1.29376318e-03], dtype=np.float64)  # Example distortion

wall_distance = 5.46  # meters

if not cap.isOpened():
    print("Error opening video file.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Store annotations: {frame_number: [(x, y)]}
annotations = {}

# Current frame number
current_frame = 0

def image_point_to_wall(u, v):
    # 1. Convert (u, v) to undistorted normalized image coordinates
    pts = np.array([[[u, v]]], dtype=np.float32)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=None)
    x_n, y_n = undistorted[0, 0]  # Normalized coords (no distortion)

    # 2. Back-project to 3D at Z = wall_distance
    X = wall_distance * x_n
    Y = wall_distance * y_n

    return float(X), float(Y)
# Click handler
def click_event(event, x, y, flags, param):
    global annotations, current_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        X, Y = image_point_to_wall(x,y)
        annotations[current_frame] = {"timestamp":current_frame / fps, "u":x, "v":y, "x":X, "y":Y}
        print(f"Annotated ({x}, {y}) at frame {current_frame} ({current_frame / fps:.2f}s)")

# Create window and set callback
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', click_event)

def show_frame(frame_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        print("Failed to load frame.")
        return None

    # Draw existing annotations
    if frame_number in annotations:
        data = annotations[frame_number]
        x = data["u"]
        y = data["v"]
        # print(x , y)
        cv2.circle(frame, (x, y), 20, (0, 0, 255), 5)

    # Show frame
    timestamp = frame_number / fps
    cv2.putText(frame, f"Frame: {frame_number}/{total_frames}  Time: {timestamp:.2f}s",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    cv2.imshow('Video', frame)

    return frame

# Main loop
while True:
    show_frame(current_frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord("a"):  # Left arrow
        current_frame = max(0, current_frame - 1)
    elif key == ord("d"):  # Right arrow
        current_frame = min(total_frames - 1, current_frame + 1)

cap.release()
cv2.destroyAllWindows()

print(annotations)

# Optional: save to file
import json

df = pd.DataFrame.from_dict(annotations, orient='index')
df.index.name = 'frame'
df.reset_index(inplace=True)
df.to_csv(f"dataset\\{name}annotations.csv", index=False)
