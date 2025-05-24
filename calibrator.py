import cv2
import numpy as np
from cv2.aruco import CharucoBoard, CharucoDetector, DICT_4X4_100
import os
import time

# configuration
squaresX = 18#11      # columns
squaresY = 11#8     # rows
squareLength = 0.01258571#0.030#0.0196363636363636 # in m
markerLength = 0.009229523#0.022#0.0144 # in m
dictionary = cv2.aruco.getPredefinedDictionary(DICT_4X4_100)
board_size = (int(2360), int(1640))  # save image size

#create charucoBoard
board = CharucoBoard(
    (squaresX, squaresY), 
    squareLength,
    markerLength,
    dictionary
)

# create and save image
board_image = board.generateImage(board_size)
cv2.imwrite("charuco_board.png", board_image)
print("saved charuco_board.png")

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images


# calibration data
all_corners = []
all_ids = []
all_object_points = [] 
all_image_points = [] 

image_size = None

images = load_images_from_folder("imgs")
for frame in images:
    # frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(frame, (1920, 1080))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = CharucoDetector(board)
    charuco_corners, charuco_ids, _, _ = detector.detectBoard(gray)
    
    # display results
    if charuco_ids is not None:
        cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
    
    cv2.imshow('Calibration', frame)
    key = cv2.waitKey(1)
    
    if charuco_ids is not None and len(charuco_ids) > 5:
        all_corners.append(charuco_corners)
        all_ids.append(charuco_ids)
        image_size = gray.shape[::-1]
        
        object_points, image_points = board.matchImagePoints(
            charuco_corners,
            charuco_ids
        )
        all_object_points.append(object_points)
        all_image_points.append(image_points)
        
        print(f"image {len(all_corners)} saved")
    else:
        print("detection failed, skipped")
    
    if key == ord('q'):
        break
    time.sleep(0.001)
    

cv2.destroyAllWindows()

# run calibration
if len(all_corners) < 15:
    print("calibration requires at least 15 images")
else:
    print("calibration started...")
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_object_points, 
        all_image_points, 
        image_size, 
        None, 
        None
    )
    
    np.savez("calibration.npz", 
             camera_matrix=camera_matrix, 
             dist_coeffs=dist_coeffs)
    
    print("\nresult saved as calibration.npz")
    print("camera matrix:\n", camera_matrix)
    print("distortion coeffs:\n", dist_coeffs.ravel())
