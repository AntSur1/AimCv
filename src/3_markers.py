import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# --- Constants ---
MARKER_SIZE = 0.06  # meters
FRONT_ID = 0
LEFT_ID = 1
RIGHT_ID = 2
REQUIRED_IDS = {FRONT_ID, LEFT_ID, RIGHT_ID}

# --- Camera plane for intersection ---
Z_PLANE = -0.025
Y_SHIFT = -0.12
Y_ZERO_OFFSET = -0.6

# --- Load camera calibration ---
data = np.load("camera_calib.npz")
CAMERA_MATRIX = data["cameraMatrix"]
DIST_COEFFS = data["distCoeffs"]

CAM_W = 640 #1280
CAM_H = 480 #720


running = True
prev_dir = np.array([0, 0, 1])

# --- ArUco setup ---
aruco = cv2.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
PARAMS = aruco.DetectorParameters()
DETECTOR = aruco.ArucoDetector(DICT, PARAMS)

# --- 3D marker model ---
OBJ_POINTS = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0],
], dtype=np.float32)


# --- Visualization setup ---

# --- Figure with GridSpec ---
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(2, 2, height_ratios=[1, 2], width_ratios=[1, 1], hspace=0.3, wspace=0.3, figure=fig)

# Top-left: 3D plot
ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_3d.set_title("3D Projection")

# Top-right: Intersection plot
ax_grid = fig.add_subplot(gs[0, 1])
ax_grid.set_title("Intersection Point on Screen")

# Bottom (spans both columns): Camera feed
ax_img = fig.add_subplot(gs[1, 1])
img_artist = ax_img.imshow(np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8))
ax_img.axis("off")
ax_img.set_title("Camera")


def on_close(event):
    global running
    running = False
    cap.release()
    cv2.destroyAllWindows()


fig.canvas.mpl_connect("close_event", on_close)
# fig.canvas.mpl_connect("key_release_event", on_close)


def update_3d_plot(marker_positions, aim_ray=None):
    ax_3d.cla()
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("3D Projection")
    ax_3d.set_xlim(-0.4, 0.4)
    ax_3d.set_ylim(-1.2, 0.4)
    ax_3d.set_zlim(-0.1, 1.2)

    for mid, p in marker_positions.items():
        x, y, z = p
        ax_3d.scatter(x, -y, z, s=10)
        ax_3d.text(x, -y, z, f"ID {mid}")

    if aim_ray is not None:
        origin, direction = aim_ray

        # compute t for intersection with Z_PLANE
        if direction[2] != 0:
            t_end = (Z_PLANE - origin[2]) / direction[2]
            t_end = max(0, t_end)  # optional: avoid negative t (behind camera)

            t = np.linspace(0, t_end, 50)
            line = origin.reshape(3,1) + direction.reshape(3,1) @ t.reshape(1,-1)
            ax_3d.plot(line[0], -line[1], line[2], color='r', linewidth=2, label='Aim')


    if True:
        # --- Screen plane size in meters ---
        screen_w = 0.525
        screen_h = 0.235*2
        x_half = screen_w / 2
        y_half = screen_h / 2

        # generate grid for plane
        x = np.linspace(-x_half, x_half, 10)
        y = np.linspace(-y_half, y_half, 10)
        X, Y = np.meshgrid(x, y)
        Z = np.full_like(X, Z_PLANE)

        # apply vertical offset (center 15cm below camera)
        Y -= Y_SHIFT + Y_ZERO_OFFSET  # or just +0.15 if you want

        # plot the surface
        ax_3d.plot_surface( X, -Y, Z, alpha=0.2, color='cyan')


    plt.draw()
    plt.pause(0.01)

def update_intersection_plot(intersection):
    ax_grid.cla()
    ax_grid.set_xlim(-0.25, 0.25)
    ax_grid.set_ylim(-0.25, 0.25)
    ax_grid.set_xlabel("X (m)")
    ax_grid.set_ylabel("Y (m)")
    ax_grid.set_title("Intersection Point on Screen")
    ax_grid.grid(True)
    if intersection is not None:
        ax_grid.scatter(-intersection[0], -intersection[1], c='r', s=50)
    fig.canvas.draw_idle()

# def estimate_aim_from_markers(Pz, Pl, Pr):
#     Zg = np.array([0, 0, 0])
#     Lg = np.array([-0.04958665,  0.00929887, -0.32420155])
#     Rg = np.array([ 0.04958665, -0.00929887, -0.29880825])

#     G = np.stack([Zg, Lg, Rg])
#     C = np.stack([Pz, Pl, Pr])

#     Gc = G - G.mean(axis=0)
#     Cc = C - C.mean(axis=0)

#     H = Gc.T @ Cc
#     U, _, Vt = np.linalg.svd(H)
#     R = Vt.T @ U.T

#     if np.linalg.det(R) < 0:
#         Vt[-1, :] *= -1
#         R = Vt.T @ U.T

#     t = C.mean(axis=0) - R @ G.mean(axis=0)

#     origin = Pz
#     direction = R @ np.array([0.0, 0.0, 1.0])
#     direction /= np.linalg.norm(direction)

#     return origin, direction, R, t



def estimate_aim_from_marker_dict(marker_positions, front_id, left_id, right_id):
    try:
        Pz = marker_positions[front_id]
        Pl = marker_positions[left_id]
        Pr = marker_positions[right_id]
    except KeyError as e:
        raise KeyError(f"Missing marker ID: {e}")

    return compute_barrel_line(Pz, Pl, Pr) #estimate_aim_from_markers(Pz, Pl, Pr)

def aim_intersection(origin, direction):
    if direction[2] == 0: return None
    t = (Z_PLANE - origin[2]) / direction[2]
    intersection = origin + t * direction
    intersection[1] += Y_SHIFT + Y_ZERO_OFFSET
    return intersection

def get_marker_positions(corners, ids):
    """Compute 3D positions of detected ArUco markers."""
    positions = {}
    for c, marker_id in zip(corners, ids.flatten()):
        ok, rvec, tvec = cv2.solvePnP(
            OBJ_POINTS,
            c[0].astype(np.float32),
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if ok:
            positions[int(marker_id)] = tvec.reshape(3)
        # print(tvec)
    return positions

# chat

def compute_barrel_line(F, B1, B2, h=0.125):
    rear_mid = 0.5 * (B1 + B2)

    s = B2 - B1
    s /= np.linalg.norm(s)

    f = F - rear_mid
    f /= np.linalg.norm(f)

    v = np.cross(f, s)
    v /= np.linalg.norm(v)

    B_virtual = rear_mid - h * v

    barrel_dir = F - B_virtual
    barrel_dir /= np.linalg.norm(barrel_dir)

    return B_virtual, barrel_dir



# --- Start webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)   # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)   # height
if not cap.isOpened(): raise RuntimeError("Could not open webcam")

last_markers = {}
last_aim = None
prev_dir = np.array([1.0, 1.0, 1.0])

print(" ============================ Running ============================ ")
while running:
    ret, frame = cap.read()
    if not ret: break
    m_frame = cv2.flip(frame, 1)

    corners, ids, _ = DETECTOR.detectMarkers(frame)
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        markers = get_marker_positions(corners, ids)
        last_markers = markers

        if REQUIRED_IDS.issubset(markers.keys()):
            origin, direction = estimate_aim_from_marker_dict(markers, FRONT_ID, LEFT_ID, RIGHT_ID)
            markers[4] = origin
            # print(origin, direction)
            alpha = 0.6
            direction_smoothed = alpha * direction + (1-alpha) * prev_dir
            direction_smoothed /= np.linalg.norm(direction_smoothed)
            prev_dir = direction_smoothed
            last_aim = (origin, direction_smoothed)

    if not last_aim is None:
        intersection = aim_intersection(*last_aim)
        update_intersection_plot(intersection)
        update_3d_plot(markers, aim_ray=(origin, direction_smoothed))

    img_artist.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

print(" =============!============ CLOSED =============!============ ")
