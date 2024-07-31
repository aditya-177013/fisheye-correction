import cv2
import numpy as np

image = cv2.imread('test2.png')

h, w = image.shape[:2]
t=(w+h)//2
camera_matrix = np.array([[t, 0, w/2],[0, t, h/2],[0, 0, 1]], dtype=np.float32)

#distortion_coefficients = np.array([-0.375, 0.425, 0, 0], dtype=np.float32)
distortion_coefficients = np.array([-0.55, 0.25, 0, 0], dtype=np.float32)

new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, distortion_coefficients, (w, h), 1, (w, h))
undistorted_image = cv2.undistort(image, camera_matrix, distortion_coefficients, None,new_camera_matrix)
x, y, w, h = roi
undistorted_image = undistorted_image[y:y+h, x:x+w]
screen_width = 700
screen_height = 400
ratio = min(screen_width / w, screen_height / h)
new_width = int(w * ratio)
new_height = int(h * ratio)
resized_image = cv2.resize(undistorted_image, (new_width, new_height))

#cv2.imwrite('undistorted_image.jpg', undistorted_image)
cv2.imshow('Undistorted Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
