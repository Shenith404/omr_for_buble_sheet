import cv2

image = cv2.imread('../images/test_23.jpeg')

alpha = 1.5  # Contrast control (1.0 = no change)
beta = 50    # Brightness control (0 = no change)

bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

cv2.imshow('Original', image)
cv2.imshow('Brightened Image', bright_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
