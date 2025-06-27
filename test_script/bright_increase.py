import cv2

# Read Image
image = cv2.imread('../images/test_6.jpeg')

# Enhance Brightness and Contrast
alpha = 1.5
beta = 50
bright_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

# Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# Adaptive Threshold
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 11, 2)

# Morphological Closing to fill gaps
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)

# Optional: Edge Detection
# edges = cv2.Canny(morph, 50, 150)

# Show Results
cv2.imshow('Original', image)
cv2.imshow('Brightened Image', bright_image)
cv2.imshow('Processed Image', thresh)
cv2.waitKey(0)
cv2.destroyAllWindows()
