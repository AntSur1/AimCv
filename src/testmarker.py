import cv2

aruco = cv2.aruco
dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

marker_id = 0
marker_size = 200  # pixels
marker = aruco.generateImageMarker(dictionary, marker_id, marker_size)

cv2.imwrite("marker_0.png", marker)
