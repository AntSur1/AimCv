import cv2
import os

cap = cv2.VideoCapture(0)
os.makedirs("calib_imgs", exist_ok=True)

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Calibration", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f"calib_imgs/img_{count}.png", frame)
        print(f"Saved image {count}")
        count += 1
    elif key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
