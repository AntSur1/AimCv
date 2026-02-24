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
Y_ZERO_OFFSET = -0.1

REAL_SCREEN_WIDTH = 0.52
REAL_SCREEN_HEIGHT = 0.29

# --- Load camera calibration ---

CAM_W =  640
CAM_H =  480

CAMERA_CALIB = np.load("src/camera_calib_64_48.npz")

# possible higher resolution requires recalibration
# CAM_W = 1280
# CAM_H = 720

running = True

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


aruco = cv2.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
PARAMS = aruco.DetectorParameters()
DETECTOR = aruco.ArucoDetector(DICT, PARAMS)
OBJ_POINTS = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0],
], dtype=np.float32)

CAMERA_MATRIX = CAMERA_CALIB["cameraMatrix"]
DIST_COEFFS = CAMERA_CALIB["distCoeffs"]

def update_3d_plot(marker_positions, origin=None, direction = None):
    """Updates and draws the intersection plot."""

    ax_3d.cla()
    ax_3d.set_xlabel("X")
    ax_3d.set_ylabel("Y")
    ax_3d.set_zlabel("Z")
    ax_3d.set_title("3D Projection")
    ax_3d.set_xlim(-0.4, 0.4)
    ax_3d.set_ylim(-1.2, 0.4)
    ax_3d.set_zlim(-0.1, 1.2)

    for id, p in marker_positions.items():
        x, y, z = p
        ax_3d.scatter(x, -y, z, s=10)
        ax_3d.text(x, -y, z, f"ID {id}")

    if origin is not None and direction is not None:
        # compute t for intersection with Z_PLANE
        if direction[2] != 0:
            t_end = (Z_PLANE - origin[2]) / direction[2]
            t_end = max(0, t_end)  # optional: avoid negative t (behind camera)

            t = np.linspace(0, t_end, 50)
            line = origin.reshape(3,1) + direction.reshape(3,1) @ t.reshape(1,-1)
            ax_3d.plot(line[0], -line[1], line[2], color='r', linewidth=2, label='Aim')


    # --- Screen plane size in meters ---
    x_half = REAL_SCREEN_WIDTH / 2
    y_half = REAL_SCREEN_HEIGHT / 2

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
    """Updates and draws the intersection plot."""
    x_half = REAL_SCREEN_WIDTH / 2
    y_half = REAL_SCREEN_HEIGHT / 2

    ax_grid.cla()
    ax_grid.set_xlim(-x_half, x_half) # actual screen size
    ax_grid.set_ylim(-y_half, y_half)
    ax_grid.set_aspect('equal', adjustable='box')
    ax_grid.set_xlabel("X (m)")
    ax_grid.set_ylabel("Y (m)")
    ax_grid.set_title("Intersection Point on Screen")
    ax_grid.grid(True)
    if intersection is not None:
        ax_grid.scatter(-intersection[0], -intersection[1], c='r', s=50)
    fig.canvas.draw_idle()

def estimate_aim_from_marker_dict(marker_positions, required_ids):
    """Helper function for compute_barrel_line."""
    front_id, left_id, right_id = required_ids
    try:
        Pz = marker_positions[front_id]
        Pl = marker_positions[left_id]
        Pr = marker_positions[right_id]
    except KeyError as e:
        raise KeyError(f"Missing marker ID: {e}")

    return compute_barrel_line(Pz, Pl, Pr)

def compute_aim_intersection(origin, direction):
    """Computes the intersection coordinte between the aim line and the aim plane."""
    if direction[2] == 0: return None
    t = (Z_PLANE - origin[2]) / direction[2]
    intersection = origin + t * direction
    intersection[1] += Y_SHIFT + Y_ZERO_OFFSET
    return intersection

def get_marker_positions(corners, ids):
    """Computes 3D positions of detected ArUco markers."""
    positions = {}
    
    for c, marker_id in zip(corners, ids.flatten()):
        ok, rvec, tvec = cv2.solvePnP(
            OBJ_POINTS,
            c[0].astype(np.float32),
            CAMERA_MATRIX,
            DIST_COEFFS,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        
        if ok: positions[int(marker_id)] = tvec.reshape(3)

    return positions

def compute_barrel_line(F, L, R, h=0.125):
    """Computes the barrel aim line based on the three Aruco codes."""
    rear_mid = 0.5 * (L + R)

    s = R - L
    s /= np.linalg.norm(s)

    f = F - rear_mid
    f /= np.linalg.norm(f)

    v = np.cross(s, f)
    v /= np.linalg.norm(v)

    B_virtual = rear_mid - h * v

    barrel_dir = F - B_virtual
    barrel_dir /= np.linalg.norm(barrel_dir)

    return B_virtual, barrel_dir

def smooth_direction(direction, prev_dir, alpha):
    """Exponential smoothing of a unit direction vector."""

    # If prev_dir is not initialized properly, initialize it
    if prev_dir is None:
        direction_norm = direction / np.linalg.norm(direction)
        return direction_norm, direction_norm

    # Exponential smoothing
    smoothed = alpha * direction + (1 - alpha) * prev_dir
    norm = np.linalg.norm(smoothed)

    # fallback safety 
    if norm == 0: return prev_dir, prev_dir

    smoothed /= norm

    return smoothed, smoothed

def detect_markers(frame):
    """Detects Aruco markers in the frame."""
    corners, ids, _ = DETECTOR.detectMarkers(frame)
    if ids is not None: aruco.drawDetectedMarkers(frame, corners, ids)
    return corners, ids

def compute_marker_aim(markers):
    """Computes 3D marker positions and the barrel aim."""
    
    if not REQUIRED_IDS.issubset(markers.keys()): return markers, None, None

    origin, direction = estimate_aim_from_marker_dict(markers, REQUIRED_IDS)
    markers[4] = origin     # Appends the back midpoint to the markers for visualization
    return markers, origin, direction

def smooth_aim(direction, prev_dir, alpha=0.6):
    """Smooths the aim direction."""
    if direction is None: return prev_dir, None

    direction_smoothed, prev_dir = smooth_direction(direction, prev_dir, alpha)
    return prev_dir, direction_smoothed

# --- Start webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)   # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)   # height
if not cap.isOpened(): raise RuntimeError("Could not open webcam")

prev_dir = None

print(" ============================ Running ============================ ")
while running:
    ret, frame = cap.read()
    if not ret: break

    corners, ids = detect_markers(frame)

    marker_coordinates = get_marker_positions(corners, ids)

    marker_coordinates, origin, direction = compute_marker_aim(marker_coordinates)

    prev_dir, smoothed_dir = smooth_aim(direction, prev_dir, alpha=0.6)

    if smoothed_dir is None: continue

    intersection = compute_aim_intersection(origin, smoothed_dir)
    update_intersection_plot(intersection)
    update_3d_plot(marker_coordinates, origin, smoothed_dir)

    img_artist.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

print(" =============!============ CLOSED =============!============ ")
