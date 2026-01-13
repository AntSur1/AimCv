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

MARKER_A_ID = 0
MARKER_B_ID = 1

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
    if ids is not None:  # at least one marker visible
        aruco.drawDetectedMarkers(frame, corners, ids)

        # --- Estimate poses ---
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, cameraMatrix, distCoeffs
        )

        # --- Get visible marker ---
        marker_id = ids.flatten()[0]  # take first visible marker
        idx = 0  # corresponding index in tvecs
        p1 = tvecs[idx].ravel()      # visible marker position

        # --- Compute forward vector using L-layout ---
        # Use a "virtual" second marker position from your rigid design
        # Example: 5.5 cm along local X axis from marker A
        L_offset = np.array([0.03, 0, 0])  # adjust to your L geometry
        p2_virtual = p1 + L_offset

        # Define the L diagonal in object coordinates (normalized)
        forward_vec = np.array([0.03, 0.03, 0])  # or [0.03, 0.03, 0] normalized
        forward_vec = forward_vec / np.linalg.norm(forward_vec)


        # --- Compute ray ---
        ray_origin = p1
        end_point_3d = ray_origin + forward_vec * 0.2

        intersection = calculatePlaneCoordinates(ray_origin, forward_vec, Z_plane)
        pts_2d, _ = cv2.projectPoints(
            np.array([ray_origin, end_point_3d]), np.zeros(3), np.zeros(3), cameraMatrix, distCoeffs
        )
        pt1, pt2 = pts_2d.reshape(2, 2).astype(int)

        # --- Draw ---
        cv2.line(frame, tuple(pt1), tuple(pt2), (255, 0, 0), 2)
        if intersection is not None:
            show_dot(intersection, dot_frame)

    cv2.imshow("Aruco Aim", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

        

cap.release()
cv2.destroyAllWindows()
