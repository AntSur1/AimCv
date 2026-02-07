import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg' if you have PyQt5 installed
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- Load camera calibration ---
data = np.load("camera_calib.npz")
cameraMatrix = data["cameraMatrix"]
distCoeffs = data["distCoeffs"]

# --- ArUco setup ---
aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
detector = aruco.ArucoDetector(dictionary, parameters)

fig = plt.figure(figsize=(10,5))
ax = fig.add_subplot(1,2,1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2)
plt.ion()


# --- Marker info ---
MARKER_SIZE = 0.06  # meters
objp = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0],
], dtype=np.float32)

FRONT_ID = 0
LEFT_ID = 1
RIGHT_ID = 2

def update_3d_view(marker_positions, aim_ray=None):
    ax.cla()
    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (up)")
    ax.set_zlabel("Z (forward)")
    ax.set_title("3d Projection")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2.0)

    # Plot markers
    for marker_id, p in marker_positions.items():
        x, y, z = p
        # flip Y for plotting
        ax.scatter(x, -y, z, s=(10/(z*z)))
        ax.text(x, -y, z, f"ID {marker_id}")

    # Plot aim line
    if aim_ray is not None:
        origin, direction = aim_ray
        d_back = 0.25  # meters behind the front marker
        length = 1.0   # meters in front of the front marker
        start_point = origin - direction * d_back

        t = np.linspace(0, length + d_back, 50)
        line = start_point.reshape(3,1) + direction.reshape(3,1) @ t.reshape(1,-1)
        # flip Y for plotting
        ax.plot(line[0], -line[1], line[2], color='r', linewidth=2, label='Aim')

    plt.draw()
    plt.pause(0.1)

def update_intersection_grid(intersection):
    ax2.cla()
    ax2.set_xlim(-2,2)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel("X (meters)")
    ax2.set_ylabel("Y (meters, up)")
    ax2.set_title("Intersection Point on Screen")
    ax2.grid(True)

    # flip Y for plotting
    ax2.scatter(-intersection[0], -intersection[1], c='r', s=50)
    fig.canvas.draw_idle()

def estimate_aim_from_markers(Pz, Pl, Pr):
    Zg = np.array([0, 0, 0])
    Lg = np.array([-0.04958665,  0.00929887, -0.32420155])
    Rg = np.array([ 0.04958665, -0.00929887, -0.29880825])

    G = np.stack([Zg, Lg, Rg])
    C = np.stack([Pz, Pl, Pr])

    Gc = G - G.mean(axis=0)
    Cc = C - C.mean(axis=0)

    H = Gc.T @ Cc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = C.mean(axis=0) - R @ G.mean(axis=0)

    origin = Pz
    direction = R @ np.array([0.0, 0.0, 1.0])
    direction /= np.linalg.norm(direction)

    return origin, direction, R, t

def estimate_aim_from_marker_dict(marker_positions, front_id, left_id, right_id):
    try:
        Pz = marker_positions[front_id]
        Pl = marker_positions[left_id]
        Pr = marker_positions[right_id]
    except KeyError as e:
        raise KeyError(f"Missing marker ID: {e}")

    return estimate_aim_from_markers(Pz, Pl, Pr)

def aim_intersection_shifted_plane(origin, direction):
    z_plane = -0.03  # plane in front of camera
    y_shift = -0.12  # adjust vertical for plotting
    y_zeroing_offset = -0.6

    if direction[2] == 0:
        return None
    t = (z_plane - origin[2]) / direction[2]
    intersection = origin + t * direction
    intersection[1] += y_shift + y_zeroing_offset
    return intersection

# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

prev_dir = np.array([0,0,1])
REQUIRED_ID = {FRONT_ID, LEFT_ID, RIGHT_ID}

while True:
    ret, frame = cap.read()
    if not ret: break
    m_frame = cv2.flip(frame, 1)

    corners, ids, _ = detector.detectMarkers(frame)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        marker_positions = {}

        for c, marker_id in zip(corners, ids.flatten()):
            imgp = c[0].astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(
                objp,
                imgp,
                cameraMatrix,
                distCoeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if ok:
                marker_positions[int(marker_id)] = tvec.reshape(3)

        if REQUIRED_ID.issubset(marker_positions.keys()):
            origin, direction, R, t = estimate_aim_from_marker_dict(marker_positions, FRONT_ID, LEFT_ID, RIGHT_ID)
            
            alpha = 0.2
            direction_smoothed = alpha * direction + (1-alpha) * prev_dir
            direction_smoothed /= np.linalg.norm(direction_smoothed)
            prev_dir = direction_smoothed

            update_3d_view(marker_positions, aim_ray=(origin, direction))
            intersection = aim_intersection_shifted_plane(origin, direction)
            update_intersection_grid(intersection)

    cv2.imshow("Aruco Aim", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
