import random
import cv2
import numpy as np


def detect_reg_number(image):
    try:
        #Apply threshold
        imgWarpGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # thresh = cv2.adaptiveThreshold(imgWarpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #                             cv2.THRESH_BINARY_INV, 11, 2)
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
        regNos = "EG_"
        #dived image into 4 equal rows
        rows = np.vsplit(erosion, 4)
        for r in rows:
            cols = np.hsplit(r, 10)
            #get the maximum white pixel box index of columns
            pixelValues = []
            for box in cols:
                totalPixels = cv2.countNonZero(box)
                print(totalPixels)
                pixelValues.append(totalPixels)
            
            #add the index of maximum white pixel box to regNos
            maxIndex = np.argmax(pixelValues)
            regNos += str(maxIndex)

        #print the registration number
        print(f"Detected Registration Number: {regNos}")
        
        return {
            "success": True,
            "digit_sequence": regNos
        }
    except :
        return {
            "success": False,
            "digit_sequence": "Error" + str(random.randint(1000,9999))
        }

