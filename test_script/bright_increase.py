import cv2
import numpy as np

import utils

# Read Image
image = cv2.imread('../images/test_24.jpg')
image = cv2.resize(image, (1025, 760))


# Grayscale
imgGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    #convert image to gray
imgBlur=cv2.GaussianBlur(imgGray,(3,3),1)       #apply blur(image_source, kernel_size, sigma)
imgCanny=cv2.Canny(imgBlur,10,50)  #apply canny edge detection (image_source, threshold1, threshold2)

 #finding all contours
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #RETR_EXTERNAL External method to find outer 
rectCon =utils.rectContour(contours)
biggestContour = utils.getCornerPoints(rectCon[0])

biggestContour=utils.reorder(biggestContour)
                
pt1=np.float32(biggestContour)
pt2=np.float32([[0,0],[1025,0],[0,760],[1025,760]])
matrix=cv2.getPerspectiveTransform(pt1,pt2)
imgWarpColored=cv2.warpPerspective(image,matrix,(1025,760))

#Apply threshold
imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(imgWarpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                               cv2.THRESH_BINARY_INV, 11, 2)

# Morphological Opening 
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
erosion =cv2.erode(thresh, kernel_open, iterations=1)



cv2.imshow('Processed Image', thresh)


# Show Results
cv2.imshow('Processed Image', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
