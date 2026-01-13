import numpy as np
import cv2

def show_dot(intersection, frame):
    # Create white frame
    dot_frame_size=(1920, 1080)
    frame = np.ones((dot_frame_size[1], dot_frame_size[0], 3), dtype=np.uint8) * 255

    # Map coordinates to pixels (simple scaling)
    scaling =  2600
    x = int(frame.shape[1] / 2 - intersection[0] * scaling)
    y = int(frame.shape[0] / 2 + intersection[1] * scaling)

    # Draw red dot
    cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

    # Show frame
    cv2.imshow("Aim Dot", frame)
    cv2.waitKey(1)  # 1ms delay
