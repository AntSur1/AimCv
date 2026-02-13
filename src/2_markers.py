import cv2
import numpy as np

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

dot_frame_size=(1920, 1080)
dot_frame = np.ones((dot_frame_size[1], dot_frame_size[0], 3), dtype=np.uint8) * 255


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# --- Marker info ---
MARKER_SIZE = 0.06  # meters 0.055
objp = np.array([
    [-MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2,  MARKER_SIZE/2, 0],
    [ MARKER_SIZE/2, -MARKER_SIZE/2, 0],
    [-MARKER_SIZE/2, -MARKER_SIZE/2, 0],
], dtype=np.float32)

Z_PLANE = -0.025

FRONT_ID = 0
REAR_ID = 1

AIM_R = np.array(
    [[ 0.28640146,  0.21225932,  0.93430198],
    [ 0.31679769, -0.94128234,  0.11673381],
    [ 0.90421979,  0.26255197, -0.3368279 ]])

# --- Helper functions ---

# def getMarkerCenter(tvec): return tvec.flatten()

# def getMarkerForwardVector(rvec):
#     R, _ = cv2.Rodrigues(rvec)
#     z_marker = np.array([0, 0, 1], dtype=np.float32).reshape(3,1)
#     forward_vector = (R @ z_marker).flatten()
#     return forward_vector

# def drawAimLine(frame, marker_center, forward_vector):
# # make sure inputs are NumPy arrays
#     marker_center = np.array(marker_center, dtype=np.float32).reshape(3)
#     forward_vector = np.array(forward_vector, dtype=np.float32).reshape(3)

#     # 3D points in camera coordinates
#     length = 0.05
#     end_point = marker_center + forward_vector * length
#     line_points_3d = np.stack([marker_center, end_point], axis=0).astype(np.float32)  # shape (2,3)

#     # project to image
#     imgpts, _ = cv2.projectPoints(
#         line_points_3d,
#         rvec=np.zeros((3,1), dtype=np.float32),
#         tvec=np.zeros((3,1), dtype=np.float32),
#         cameraMatrix=cameraMatrix,
#         distCoeffs=distCoeffs
#     )

#     # flatten and cast to int
#     pt0 = tuple(map(int, imgpts[0,0]))
#     pt1 = tuple(map(int, imgpts[1,0]))

#     # draw the line
#     cv2.line(frame, pt0, pt1, (255, 50, 0), 3)

# def getAimIntersectionPoint(marker_center, forward_vector):
#     plane_point = np.array([0, 0, -0.1])
#     plane_normal = np.array([0, 0, 1])

#     denom = np.dot(plane_normal, forward_vector)
#     if np.abs(denom) < 1e-3:
#         # Line is parallel to plane, no single intersection
#         intersection = None
#     else:
#         t = np.dot(plane_normal, plane_point - marker_center) / denom
#         intersection = marker_center + forward_vector * t
#     return intersection

# def show_dot_frame(dot_history):
#     # Create white frame
#     frame = np.ones((dot_frame_size[1], dot_frame_size[0], 3), dtype=np.uint8) * 255

#     for d in dot_history:
#         # intersection_point = [x, y, z] in meters
#         scale = 1000
#         x = int(dot_frame_size[0] / 2 - d[0] * scale)
#         y = int(dot_frame_size[1] / 2 + d[1] * scale)


#         # Draw red dot
#         #print("in", d, "print", x, y)
#         cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

#     # Show frame
#     cv2.imshow("Aim Dot", frame)
#     cv2.waitKey(1)  # 1ms delay

# def project_to_screen(intersection):
#     if intersection is None:
#         return None

#     imgpts, _ = cv2.projectPoints(
#         intersection.reshape(1,3),
#         rvec=np.zeros((3,1), dtype=np.float32),
#         tvec=np.zeros((3,1), dtype=np.float32),
#         cameraMatrix=cameraMatrix,
#         distCoeffs=distCoeffs
#     )
#     x, y = imgpts[0,0]

#     # Only clamp if within reasonable range
#     if 0 <= x < dot_frame_size[0] and 0 <= y < dot_frame_size[1]:
#         return int(x), int(y)
#     else:
#         return None  # ignore points outside view

# def intersect_line_plane(p_line, d_line,):
#     """
#     p_line: np.array([x,y,z]) start of line
#     d_line: np.array([dx,dy,dz]) unit direction vector
#     p_plane: np.array([x,y,z]) point on plane
#     n_plane: np.array([nx,ny,nz]) plane normal
#     """
#     z_screen = 0

#     p_plane = np.array([0, 0, z_screen])
#     n_plane = np.array([0, 0, 1])
#     denom = np.dot(d_line, n_plane)
#     if abs(denom) < 1e-6:
#         return None  # line parallel to plane

#     t = np.dot(p_plane - p_line, n_plane) / denom
#     intersection = p_line + t * d_line
#     return intersection




def setup_plot(ax):
    
    plt.ion()  # interactive mode


def update_3d_view(marker_positions, d_marker):
    ax.cla()

    ax.set_xlabel("X (right)")
    ax.set_ylabel("Y (down)")
    ax.set_zlabel("Z (forward)")

    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_zlim(0, 2.0)

     # Plot markers
    for marker_id, p in marker_positions.items():
        x, y, z = p
        ax.scatter(x, y, z, s=60/z)
        ax.text(x, y, z, f"ID {marker_id}")

    # Plot marker axis 
    if REAR_ID in marker_positions:
        L = 0.4 
        p_rear = marker_positions[REAR_ID]
        p2 = p_rear + L * d_marker

        ax.plot(
            [p_rear[0], p2[0]],
            [p_rear[1], p2[1]],
            [p_rear[2], p2[2]],
            color='red',
            linewidth=2,
            label='marker axis'
        )

        #Plot aim line
        d_aim = AIM_R @ d_marker
        d_aim /= np.linalg.norm(d_aim)  # ensure unit vector

        p_aim_origin = marker_positions[FRONT_ID]


        p2_aim = p_aim_origin - L * d_aim
        ax.plot(
            [p_aim_origin[0], p2_aim[0]],
            [p_aim_origin[1], p2_aim[1]],
            [p_aim_origin[2], p2_aim[2]],
            color='green',
            linewidth=2,
            label='true aim line'
        )


        
        h_point = intersect_line_plane(p_aim_origin, d_aim)
        ax.scatter(h_point[0], h_point[1], h_point[2], 
           color='blue', s=60, label='aim hit')

    plt.draw()
    plt.pause(0.0001)


# --- Start webcam ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam")

setup_plot(ax)

dot_history = []

debug_tick = 0
while True:
    debug_tick += 0
    ret, frame = cap.read()
    m_frame = cv2.flip(frame, 1)
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)
    
    if ids is not None: 
        aruco.drawDetectedMarkers(frame, corners, ids)

        # --- Estimate poses ---
        marker_positions = {}

        for c , marker_id in zip(corners, ids.flatten()):
            imgp = c[0].astype(np.float32)  # (4,2)

            ok, rvec, tvec = cv2.solvePnP(
                objp,
                imgp,
                cameraMatrix,
                distCoeffs,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )

            if ok: marker_positions[int(marker_id)] = tvec.reshape(3)

        if REAR_ID in marker_positions and FRONT_ID in marker_positions:
            p_rear = marker_positions[REAR_ID]
            p_front = marker_positions[FRONT_ID]

            # Marker axis vector
            d_marker = p_front - p_rear
            norm = np.linalg.norm(d_marker)
            if norm > 1e-6:
                d_marker /= norm

                print(marker_positions, d_marker)
                
                update_3d_view(marker_positions, d_marker)


    cv2.imshow("Aruco Aim", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

        

cap.release()
cv2.destroyAllWindows()
