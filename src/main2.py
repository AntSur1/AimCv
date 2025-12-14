import cv2
import numpy as np

# --- Load camera calibration ---
data = np.load("camera_calib.npz")
cameraMatrix = data["cameraMatrix"]
distCoeffs = data["distCoeffs"]

# --- ArUco setup ---
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

# --- Marker info ---
MARKER_SIZE = 0.055  # meters

# --- Virtual screen plane ---
SCREEN_Z = 0.1

def intersect_ray_plane(origin, direction, plane_z):
    """Return intersection point of a ray with plane z=plane_z"""
    t = (plane_z - origin[2]) / direction[2]
    if t < 0:
        return 0
    return origin + t * direction

# --- Smoothing parameters ---
alpha_pos = 0.3  # smoothing factor for position
alpha_dir = 0.3  # smoothing factor for direction

# Initialize smoothed values
smoothed_origin = None
smoothed_forward = None

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

        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, 0.03)

            # Compute forward vector
            R, _ = cv2.Rodrigues(rvec)
            forward_vec = R[:, 2]
            forward_vec /= np.linalg.norm(forward_vec)

            ray_origin = tvec.ravel()

            # --- Apply exponential smoothing ---
            if smoothed_origin is None:
                smoothed_origin = ray_origin
            else:
                smoothed_origin = alpha_pos * ray_origin + (1 - alpha_pos) * smoothed_origin

            if smoothed_forward is None:
                smoothed_forward = forward_vec
            else:
                smoothed_forward = alpha_dir * forward_vec + (1 - alpha_dir) * smoothed_forward
                smoothed_forward /= np.linalg.norm(smoothed_forward)

            # Compute aim point on screen plane
            # if abs(forward_vec[2]) < 1e-6:
            #     aim_point = None  # forward vector nearly parallel to plane
            # else:
            #     aim_point = intersect_ray_plane(smoothed_origin, smoothed_forward, SCREEN_Z)
            
            # if aim_point is not None and np.isfinite(aim_point).all():
            #     pts_2d, _ = cv2.projectPoints(
            #         np.array([aim_point], dtype=np.float32).reshape(-1,3),
            #         np.zeros(3),
            #         np.zeros(3),
            #         cameraMatrix,
            #         distCoeffs
            #     )
            #     pt = (int(pts_2d[0,0,0]), int(pts_2d[0,0,1]))
            #     cv2.circle(frame, pt, 8, (0,0,255), -1)

            

            # Draw smoothed forward ray
            end_point_3d = smoothed_origin + smoothed_forward * 0.1
            pts_2d, _ = cv2.projectPoints(
                np.array([smoothed_origin, end_point_3d]), np.zeros(3), np.zeros(3), cameraMatrix, distCoeffs
            )
            pt1, pt2 = pts_2d.reshape(2, 2).astype(int)
            cv2.line(frame, tuple(pt1), tuple(pt2), (255, 0, 0), 2)  # blue line

    cv2.imshow("Aruco Aim Demo - Smoothed", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
