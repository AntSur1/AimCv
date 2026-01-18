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

dot_frame_size=(1920, 1080)
dot_frame = np.ones((dot_frame_size[1], dot_frame_size[0], 3), dtype=np.uint8) * 255

# --- Marker info ---
MARKER_SIZE = 0.06  # meters 0.055
objp = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0],
], dtype=np.float32)

Z_PLANE = -0.025

MARKER_A_ID = 0
MARKER_B_ID = 1

# --- Helper functions ---

def getMarkerCenter(tvec): return tvec.flatten()

def getMarkerForwardVector(rvec):
    R, _ = cv2.Rodrigues(rvec)
    z_marker = np.array([0, 0, 1], dtype=np.float32).reshape(3,1)
    forward_vector = (R @ z_marker).flatten()
    return forward_vector

def drawAimLine(frame, marker_center, forward_vector):
# make sure inputs are NumPy arrays
    marker_center = np.array(marker_center, dtype=np.float32).reshape(3)
    forward_vector = np.array(forward_vector, dtype=np.float32).reshape(3)

    # 3D points in camera coordinates
    length = 0.05
    end_point = marker_center + forward_vector * length
    line_points_3d = np.stack([marker_center, end_point], axis=0).astype(np.float32)  # shape (2,3)

    # project to image
    imgpts, _ = cv2.projectPoints(
        line_points_3d,
        rvec=np.zeros((3,1), dtype=np.float32),
        tvec=np.zeros((3,1), dtype=np.float32),
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs
    )

    # flatten and cast to int
    pt0 = tuple(map(int, imgpts[0,0]))
    pt1 = tuple(map(int, imgpts[1,0]))

    # draw the line
    cv2.line(frame, pt0, pt1, (255, 50, 0), 3)

def getAimIntersectionPoint(marker_center, forward_vector):
    plane_point = np.array([0, 0, -0.1])
    plane_normal = np.array([0, 0, 1])

    denom = np.dot(plane_normal, forward_vector)
    if np.abs(denom) < 1e-3:
        # Line is parallel to plane, no single intersection
        intersection = None
    else:
        t = np.dot(plane_normal, plane_point - marker_center) / denom
        intersection = marker_center + forward_vector * t
    return intersection


def show_dot_frame(dot_history):
    # Create white frame
    frame = np.ones((dot_frame_size[1], dot_frame_size[0], 3), dtype=np.uint8) * 255

    for d in dot_history:
        # intersection_point = [x, y, z] in meters
        scale = 1000
        x = int(dot_frame_size[0] / 2 - d[0] * scale)
        y = int(dot_frame_size[1] / 2 + d[1] * scale)


        # Draw red dot
        #print("in", d, "print", x, y)
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Show frame
    cv2.imshow("Aim Dot", frame)
    cv2.waitKey(1)  # 1ms delay

def project_to_screen(intersection):
    if intersection is None:
        return None

    imgpts, _ = cv2.projectPoints(
        intersection.reshape(1,3),
        rvec=np.zeros((3,1), dtype=np.float32),
        tvec=np.zeros((3,1), dtype=np.float32),
        cameraMatrix=cameraMatrix,
        distCoeffs=distCoeffs
    )
    x, y = imgpts[0,0]

    # Only clamp if within reasonable range
    if 0 <= x < dot_frame_size[0] and 0 <= y < dot_frame_size[1]:
        return int(x), int(y)
    else:
        return None  # ignore points outside view



# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")


dot_history = []
while True:
    ret, frame = cap.read()
    m_frame = cv2.flip(frame, 1)
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)
    
    if ids is not None: 
        aruco.drawDetectedMarkers(frame, corners, ids)

        # --- Estimate poses ---
        rvecs = []
        tvecs = []

        for c in corners:
            imgp = c[0].astype(np.float32)  # (4,2)

            ok, rvec, tvec = cv2.solvePnP(
                objp,
                imgp,
                cameraMatrix,
                distCoeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if ok:
                rvecs.append(rvec)
                tvecs.append(tvec)
        
        for rvec, tvec in zip(rvecs, tvecs):
            cv2.drawFrameAxes(frame, cameraMatrix, distCoeffs, rvec, tvec, MARKER_SIZE * 0.5, 3)
           
            marker_center = getMarkerCenter(tvec)
            forward_vector = getMarkerForwardVector(rvec)

            drawAimLine(frame, marker_center, forward_vector)
            intersection_point = getAimIntersectionPoint(marker_center, forward_vector)
            print(intersection_point)

            if intersection_point is not None:
                dot_history.append(intersection_point)

            if len(dot_history) > 20: 
                dot_history.pop(0)

            show_dot_frame(dot_history)
            


    cv2.imshow("Aruco Aim", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

        

cap.release()
cv2.destroyAllWindows()
