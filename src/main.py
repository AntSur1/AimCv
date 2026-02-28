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
Y_SHIFT = -0.275

REAL_SCREEN_WIDTH = 0.53
REAL_SCREEN_HEIGHT = 0.29
PX_SCREEN_WIDTH  = 1920
PX_SCREEN_HEIGHT = 1080

SCALE_X = PX_SCREEN_WIDTH  / REAL_SCREEN_WIDTH
SCALE_Y = PX_SCREEN_HEIGHT / REAL_SCREEN_HEIGHT
CENTER_X = PX_SCREEN_WIDTH  / 2
CENTER_Y = PX_SCREEN_HEIGHT / 2

# --- Camera calibration ---
CAMERA_CALIB = np.load("src/camera_calib_128_72.npz")
CAM_W = 1280
CAM_H = 720

CAMERA_MATRIX = CAMERA_CALIB["cameraMatrix"]
DIST_COEFFS = CAMERA_CALIB["distCoeffs"]

# --- ArUco detector ---
aruco = cv2.aruco
DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
PARAMS = aruco.DetectorParameters()
DETECTOR = aruco.ArucoDetector(DICT, PARAMS)

# --- 3D marker object points ---
OBJ_POINTS = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0],
], dtype=np.float32)

# --- Global flags ---
running = True
DEBUG_MODE = False
print("Debug mode:", DEBUG_MODE)

# --- Figure and axes ---
fig = plt.figure(figsize=(16, 9))
gs = GridSpec(1, 2, hspace=0.3, wspace=0.3, figure=fig)

ax_3d = fig.add_subplot(gs[0, 0], projection='3d')
ax_3d.set_title("3D Projection")
ax_3d.set_xlabel("X")
ax_3d.set_ylabel("Y")
ax_3d.set_zlabel("Z")
ax_3d.set_xlim(-0.4, 0.4)
ax_3d.set_ylim(-1.2, 0.4)
ax_3d.set_zlim(-0.1, 1.2)

ax_grid = fig.add_subplot(gs[0, 1])
ax_grid.set_title("Intersection Point on Screen")

# x_half = REAL_SCREEN_WIDTH / 2
# y_half = REAL_SCREEN_HEIGHT / 2
# ax_grid.set_xlim(-x_half, x_half)
# ax_grid.set_ylim(-y_half, y_half)

ax_grid.set_xlim(0, PX_SCREEN_WIDTH)
ax_grid.set_ylim(PX_SCREEN_HEIGHT, 0)

ax_grid.set_aspect('equal', adjustable='box')
ax_grid.grid(True)

ax_grid.grid(True)
intersection_scat = ax_grid.scatter([], [], c='r', s=50)

# --- Pre-create 3D plot elements ---
marker_scat = ax_3d.scatter([], [], [], s=10)
aim_line, = ax_3d.plot([], [], [], color='r', linewidth=2, label='Aim')

# Static screen plane
x = np.linspace(-REAL_SCREEN_WIDTH / 2,REAL_SCREEN_WIDTH / 2, 10)
y = np.linspace(-REAL_SCREEN_HEIGHT / 2, REAL_SCREEN_HEIGHT / 2, 10)
X, Y = np.meshgrid(x, y)
Z = np.full_like(X, Z_PLANE)
Y += Y_SHIFT
ax_3d.plot_surface(X, Y, Z, alpha=0.2, color='cyan')

print("Variables set")

def end_program(event=None):
    print(" Closing program")
    global running
    running = False


fig.canvas.mpl_connect("close_event", end_program)
# fig.canvas.mpl_connect("key_release_event", end_program)

def update_3d_plot(marker_positions, origin=None, direction=None):
    '''Updates the 3D plot.'''
    # Update marker positions
    if marker_positions:
        xs, ys, zs = zip(*marker_positions.values())
        ids = list(marker_positions.keys())
        
        # Use IDs to assign colors (cycled through a colormap)
        cmap = plt.get_cmap('tab10')
        colors = [cmap(i % 10) for i in ids]

        marker_scat._offsets3d = (xs, [-y for y in ys], zs)
        marker_scat.set_color(colors)

        # Remove previous text labels safely
        for txt in ax_3d.texts:
            txt.remove()

        # Draw new text labels
        for id, (x, y, z) in marker_positions.items():
            ax_3d.text(x, -y, z, str(id), color=cmap(id % 10))
    else:
        marker_scat._offsets3d = ([], [], [])

    # Update aim line
    if origin is not None and direction is not None and direction[2] != 0:
        t_end = max(0, (Z_PLANE - origin[2]) / direction[2])
        t = np.linspace(0, t_end, 50)
        line = origin.reshape(3,1) + direction.reshape(3,1) @ t.reshape(1,-1)
        aim_line.set_data(line[0], -line[1])
        aim_line.set_3d_properties(line[2])

    # Redraw
    fig.canvas.draw_idle()
    fig.canvas.flush_events()

def update_intersection_plot(intersection):
    '''Updates the intersection plot.'''
    if intersection is not None:
        intersection_scat.set_offsets([[-intersection[0], -intersection[1]]])
    else:
        intersection_scat.set_offsets(np.empty((0, 2)))
    fig.canvas.draw_idle()

def plot_screen_px_plot(coords):
    if coords is not None:
        x, y = coords
        intersection_scat.set_offsets([[x, y]])
    else:
        intersection_scat.set_offsets(np.empty((0, 2)))
    
    fig.canvas.draw_idle()

def update_camera_frame(frame, are_all_markers_in_frame, display_size=(320, 180)):
    """Updates the camera feed image in the matplotlib UI."""
    # Resize for display only
    frame_resized = cv2.resize(frame, display_size)
    h, w = frame_resized.shape[:2]
    # (B, G, R) for some reason
    color = (0, 255, 0) if are_all_markers_in_frame else (0, 0, 255)

    THICKNESS = 4
    cv2.rectangle(frame_resized, (0, h-THICKNESS), (w, h), color, -1)
    
    # Update window    
    cv2.resizeWindow("Camera", display_size[0], display_size[1])
    cv2.imshow("Camera", frame_resized)

    # Optional: close program on 'Esc' key
    if (cv2.getWindowProperty("Camera", cv2.WND_PROP_VISIBLE) < 1 or 
        (cv2.waitKey(1) & 0xFF == 27)): 
        end_program()

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
    intersection[1] += Y_SHIFT
    return intersection

def intersection_to_screen_coordinates(intersection):
    x = CENTER_X - SCALE_X * intersection[0]
    y = CENTER_Y + SCALE_Y * intersection[1]
    x = round(x)
    y = round(y)
    return x, y

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
    rear_mid = 0.5 * (R + L)

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

def smooth_aim(direction, prev_dir, alpha=1):
    """Smooths the aim direction."""
    if direction is None: return prev_dir, None

    direction_smoothed, prev_dir = smooth_direction(direction, prev_dir, alpha)
    return prev_dir, direction_smoothed

# --- Start webcam ---
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)   # width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)   # height
if not cap.isOpened(): raise RuntimeError("Could not open webcam")
print("Camnera loaded")

cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)

if DEBUG_MODE:
    plt.ion()
    plt.show()

prev_dir = None

print(" ============================ Running ============================ ")
while running:
    ret, frame = cap.read()
    if not ret: break

    corners, ids = detect_markers(frame)
    can_see_all_markers = (
        ids is not None and
        REQUIRED_IDS.issubset(set(ids.flatten()))
    )
    
    update_camera_frame(frame, can_see_all_markers)

    if ids is None or len(ids) == 0: continue

    # --- Get 3D positions for all detected markers ---
    marker_coordinates = get_marker_positions(corners, ids)

    # --- Compute aim and intersection only if required markers exist ---
    smoothed_dir = intersection = origin = screen_coords = None

    if can_see_all_markers:
        origin, direction = estimate_aim_from_marker_dict(marker_coordinates, REQUIRED_IDS)
        prev_dir, smoothed_dir = smooth_aim(direction, prev_dir, 0.3)
        intersection = compute_aim_intersection(origin, smoothed_dir)
        screen_coords = intersection_to_screen_coordinates(intersection)

    # --- Update visualizations ---
    if DEBUG_MODE: 
        # update_intersection_plot(intersection)
        plot_screen_px_plot(screen_coords)
        update_3d_plot(marker_coordinates, origin, smoothed_dir)

print(" =============!============ CLOSED =============!============ ")
cap.release()
cv2.destroyAllWindows()
