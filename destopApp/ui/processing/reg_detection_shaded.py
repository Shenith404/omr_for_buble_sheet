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

       

        regNos = "EG_"
        #dived image into 4 equal rows
        rows = np.vsplit(thresh, 4)
        for r in rows:
            cols = np.hsplit(r, 10)
            #get the maximum white pixel box index of columns
            pixelValues = []
            for box in cols:
                h, w = box.shape
                left = int(w * 0.2)
                right = int(w * 0.8)
                top = int(h * 0.2)
                bottom = int(h * 0.8)
                box = box[top:bottom, left:right]
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

