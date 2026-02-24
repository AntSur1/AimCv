import cv2
import numpy as np
import glob

# --- chessboard settings ---
CHESSBOARD_SIZE = (9, 6)   # inner corners
SQUARE_SIZE = 0.024        # meters at 50%

CAM_W =  1280   #640#
CAM_H =  720    #480#
name = "128_72"

# --- prepare object points ---
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []
imgpoints = []


images = glob.glob("calib_imgs/*.png")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret:
        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        objpoints.append(objp)
        imgpoints.append(corners)
        cv2.drawChessboardCorners(img, CHESSBOARD_SIZE, corners, ret)
        cv2.imshow("Corners", img)
        cv2.waitKey(500)

cv2.destroyAllWindows()

# --- calibrate ---
def compute_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i], rvecs[i], tvecs[i], mtx, dist
        )
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
    return total_error / len(objpoints)


image_size = gray.shape[::-1]

# --- Full model ---
ret_full, mtx_full, dist_full, rvecs_full, tvecs_full = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None
)

err_full = compute_error(objpoints, imgpoints,
                         rvecs_full, tvecs_full,
                         mtx_full, dist_full)

# --- k3 fixed model ---
flags = cv2.CALIB_FIX_K3

ret_k3, mtx_k3, dist_k3, rvecs_k3, tvecs_k3 = cv2.calibrateCamera(
    objpoints,
    imgpoints,
    image_size,
    None,
    None,
    flags=flags
)

err_k3 = compute_error(objpoints, imgpoints,
                       rvecs_k3, tvecs_k3,
                       mtx_k3, dist_k3)

# ----------------------------
# Results
# ----------------------------

print("\n===== FULL MODEL =====")
print("RMS:", ret_full)
print("Mean reprojection error:", err_full)
print("Camera matrix:\n", mtx_full)
print("Distortion:\n", dist_full)

print("\n===== K3 FIXED =====")
print("RMS:", ret_k3)
print("Mean reprojection error:", err_k3)
print("Camera matrix:\n", mtx_k3)
print("Distortion:\n", dist_k3)

# ----------------------------
# Choose best model automatically
# ----------------------------

if err_k3 <= err_full + 0.05:
    print("\nUsing K3-FIXED model")
    cameraMatrix = mtx_k3
    distCoeffs = dist_k3
else:
    print("\nUsing FULL model")
    cameraMatrix = mtx_full
    distCoeffs = dist_full

np.savez(f"camera_calib_{name}.npz",
         cameraMatrix=cameraMatrix,
         distCoeffs=distCoeffs)

print("\nCalibration saved.")
