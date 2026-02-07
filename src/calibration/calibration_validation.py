import cv2
import numpy as np

data = np.load("camera_calib.npz")
cameraMatrix = data["cameraMatrix"]
distCoeffs = data["distCoeffs"]

cap = cv2.VideoCapture(0)

ret, frame = cap.read()
h, w = frame.shape[:2]

newCameraMatrix, roi = cv2.getOptimalNewCameraMatrix(
    cameraMatrix,
    distCoeffs,
    (w, h),
    1
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    undistorted = cv2.undistort(
        frame,
        cameraMatrix,
        distCoeffs,
        None,
        newCameraMatrix
    )

    both = cv2.hconcat([frame, undistorted])
    cv2.imshow("Original | Undistorted", both)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
print(distCoeffs.ravel())

