import cv2
import numpy as np
from display_dot import show_dot

# --- Load camera calibration ---
data = np.load("camera_calib.npz")
cameraMatrix = data["cameraMatrix"]
distCoeffs = data["distCoeffs"]

# --- ArUco setup ---
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

dot_frame_size=(1920, 1080)
dot_frame = np.ones((dot_frame_size[1], dot_frame_size[0], 3), dtype=np.uint8) * 255

# --- Marker info ---
MARKER_SIZE = 0.06  # meters 0.055
Z_plane = -0.025


# --- Smoothing ---
alpha_dir = 0.4
filtered_dirs = {}
last_forward_vecs = {}



# --- Helper functions ---

def calculatePlaneCoordinates(origin, direction, z_plane):
    # ray: p = ray_origin + t * forward_vec
    if abs(direction[2]) < 1e-6:
        return None

    t = (z_plane - origin[2]) / direction[2]
    intersection = origin + direction * t
    return tuple(intersection)


# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, cameraMatrix, distCoeffs
        )

        for rvec, tvec, marker_id in zip(rvecs, tvecs, ids.flatten()):
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)

            # Compute forward vector
            R, _ = cv2.Rodrigues(rvec)
            forward_vec = R[:, 2]
            forward_vec /= np.linalg.norm(forward_vec)

            marker_id = int(marker_id)

            if marker_id not in filtered_dirs:
                filtered_dirs[marker_id] = forward_vec.copy()
                last_forward_vecs[marker_id] = forward_vec.copy()
            else:
                # dead-zone
                if abs(forward_vec[0]) < 0.01 and abs(forward_vec[1]) < 0.01:
                    forward_vec = last_forward_vecs[marker_id].copy()

                last_forward_vecs[marker_id] = forward_vec.copy()

                # filtering
                filtered_dirs[marker_id] = (
                    alpha_dir * forward_vec +
                    (1 - alpha_dir) * filtered_dirs[marker_id]
                )
                filtered_dirs[marker_id] /= np.linalg.norm(filtered_dirs[marker_id])

            filtered_dir = filtered_dirs[marker_id]

            ray_origin = tvec.ravel()
            end_point_3d = ray_origin + filtered_dir * 0.2

            intersection = calculatePlaneCoordinates(ray_origin, filtered_dir, Z_plane)
            pts_2d, _ = cv2.projectPoints(
                np.array([ray_origin, end_point_3d]), np.zeros(3), np.zeros(3), cameraMatrix, distCoeffs
            )
            pt1, pt2 = pts_2d.reshape(2, 2).astype(int)
            
            
            if intersection is not None:
                cv2.line(frame, tuple(pt1), tuple(pt2), (255, 0, 0), 2)  # blue line
                show_dot(intersection, dot_frame)
    

    cv2.imshow("Aruco Aim", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break
        

cap.release()
cv2.destroyAllWindows()
