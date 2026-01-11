import cv2
import numpy as np

import utils

def detect_reg_number(image):
   
    #Apply threshold
    imgWarpGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(imgWarpGray, 100, 255, cv2.THRESH_BINARY_INV)

    # Morphological Opening 
    kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    erosion =cv2.erode(thresh, kernel_open, iterations=1)

    # Crop image: 1% from left/right, 5% from top/bottom
    h, w = erosion.shape
    left = int(w * 0.01)
    right = int(w * 0.99)
    top = int(h * 0.05)
    bottom = int(h * 0.95)
    erosion = erosion[top:bottom, left:right]

    cv2.imshow("Eroded Image", erosion)
    cv2.waitKey(0)
    regNos = "EG_"
    #dived image into 4 equal rows
    rows = np.vsplit(erosion, 4)
    for r in rows:
        cols = np.hsplit(r, 10)
        #get the maximum white pixel box index of columns
        pixelValues = []
        for box in cols:
             # Crop image: 1% from left/right, 5% from top/bottom
            h, w = box.shape
            left = int(w * 0.1)
            right = int(w * 0.9)
            top = int(h * 0.1)
            bottom = int(h * 0.9)
            box = box[top:bottom, left:right]
            totalPixels = cv2.countNonZero(box)
            print(totalPixels)
            pixelValues.append(totalPixels)
        
        #add the index of maximum white pixel box to regNos
        maxIndex = np.argmax(pixelValues)
        regNos += str(maxIndex)

    #print the registration number
    print(f"Detected Registration Number: {regNos}")
    
    return regNos

