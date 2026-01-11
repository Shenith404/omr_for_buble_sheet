import cv2
import numpy as np

import utils
import reg_detection_shaded as r

# Read Image
image = cv2.imread('../images/EG_1278.png')
image = cv2.resize(image, (1025, 760))


# Grayscale
imgGray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)    #convert image to gray
imgBlur=cv2.GaussianBlur(imgGray,(3,3),1)       #apply blur(image_source, kernel_size, sigma)
imgCanny=cv2.Canny(imgBlur,10,50)  #apply canny edge detection (image_source, threshold1, threshold2)

 #finding all contours
contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #RETR_EXTERNAL External method to find outer 
rectCon =utils.rectContour(contours)
print(len(rectCon))
biggestContour = utils.getCornerPoints(rectCon[0])

for x in rectCon:
    print(cv2.contourArea(x))


#read registration number
if len(rectCon) > 1:  # Check if we have at least 2 rectangles before accessing rects[1]
    reg_contour = utils.getCornerPoints(rectCon[1]) 

    if reg_contour.size != 0:
        reg_contour = utils.reorder(reg_contour)
        pts1r = np.float32(reg_contour)
        pts2r = np.float32([[0, 0], [1000, 0], [0, 400], [1000, 400]])
        matrix = cv2.getPerspectiveTransform(pts1r, pts2r)
        warped_r = cv2.warpPerspective(image, matrix, (1000, 400))
        
        try:
            reg_number =  r.detect_reg_number(warped_r)
            print("Reg Number Detection: ", reg_number)

            if reg_number["success"]:
                # Rename the file as registration number
                reg_num_str = reg_number["digit_sequence"]
                if reg_num_str:  # Check if digit_sequence is not empty
                    new_filename = f"{reg_num_str}.png"
                else:
                    print("Warning: Registration number detected but digit sequence is empty")
            else:
                print("Warning: Registration number detection failed or no digits detected")
        except Exception as e:
            print(f"Error during registration number detection: {str(e)}")
    else:
        print("Warning: Registration contour not found or invalid")
else:
    print("Warning: Insufficient rectangles detected for registration number processing")
          


    
biggestContour=utils.reorder(biggestContour)

                
pt1=np.float32(biggestContour)
pt2=np.float32([[0,0],[1025,0],[0,760],[1025,760]])
matrix=cv2.getPerspectiveTransform(pt1,pt2)
imgWarpColored=cv2.warpPerspective(image,matrix,(1025,760))

#Apply threshold
imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

thresh = cv2.adaptiveThreshold(imgWarpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)
#_, thresh = cv2.threshold(imgWarpGray, 125, 255, cv2.THRESH_BINARY_INV)

# Morphological Opening 
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
erosion =cv2.erode(thresh, kernel_open, iterations=1)



cv2.imshow('Processed Image', thresh)


# Show Results
cv2.imshow('Processed Image', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
