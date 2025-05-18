import os
import time
import uuid
import numpy as np
import cv2

def stackImages(imgArray,scale,lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    newWidth = int(width * scale)
    newHeight = int(height * scale)
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (newWidth, newHeight))
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (newWidth, newHeight))
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth= int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        print(eachImgHeight)
        for d in range(0, rows):
            for c in range (0,cols):
                cv2.rectangle(ver,(c*eachImgWidth,eachImgHeight*d),(c*eachImgWidth+len(lables[d])*13+27,30+eachImgHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lables[d],(eachImgWidth*c+10,eachImgHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def rectContour(contours):
    rectCon = []
    for i in contours:
        area = cv2.contourArea(i)
        if area > 50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if len(approx) == 4:
                rectCon.append(i)
    rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
    return rectCon

def getCornerPoints(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx

def reorder(myPoints):
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    add = myPoints.sum(1)
   # print(myPoints)
    #print(add)
    #smallest one will be top left
    myPointsNew[0] = myPoints[np.argmin(add)]
    #largest one will be bottom right
    myPointsNew[3] = myPoints[np.argmax(add)]
    diff = np.diff(myPoints, axis=1)
    #second smallest will be top right
    myPointsNew[1] = myPoints[np.argmin(diff)]
    #second largest will be bottom left
    myPointsNew[2] = myPoints[np.argmax(diff)]

    return myPointsNew


# def splitBoxes(img):
#     rows = np.vsplit(img,10)
#     cv2.imshow('Split',rows[0])
#     boxes = []
#     for r in rows:
#         cols= np.hsplit(r,4)
#         cv2.imshow('Spl4it',cols[0])
#         for box in cols:
#             boxes.append(box)

#     return boxes


#get one  answer box from the image
def verticalSplitBoxes(img):
    #split image into 5 columns
    rows = np.hsplit(img,5)
    cv2.imshow('Split',rows[0])
    boxes = []
    for r in rows:
        #split each column into 10 answer rows
        cols= np.vsplit(r,10)
        #cv2.imshow('Spl4it',cols[9])
        for box in cols:
            #add each answer box to the list
           # cv2.imshow('Spl4it',box)
            boxes.append(box)
    return boxes

#divide answer boxes into 7 parts
def getAnswerBlocks(img):
    #reshape the image 
    img=cv2.resize(img,(350,50))
    #cv2.imshow('blocks',img)

    blocks = np.hsplit(img,7)
    boxes = []
    for block in blocks:
        
        boxes.append(block)
    return boxes


#safe images in new folder
def saveImages(answerBlocks, ImageName, boxNumber):
    # Ensure directories exist
    save_path_cross_Images = r"E:\University\fyp\mcq_test_1\cross_Images"
    save_path_empty_Images = r"E:\University\fyp\mcq_test_1\empty_Images"
    os.makedirs(save_path_cross_Images, exist_ok=True)
    os.makedirs(save_path_empty_Images, exist_ok=True)

    for i, block in enumerate(answerBlocks):
        totalPixels = cv2.countNonZero(block)
        
        # Generate a unique filename using timestamp + UUID
        unique_id = str(uuid.uuid4())[:8]  # First 8 characters of UUID
        timestamp = int(time.time())  # Current timestamp
        filename = f"{ImageName}_{boxNumber + i}_{timestamp}_{unique_id}.jpg"

        if totalPixels < 550:  # Empty image condition
            file_path = os.path.join(save_path_empty_Images, filename)
        else:  # Marked image condition
            file_path = os.path.join(save_path_cross_Images, filename)

        cv2.imwrite(file_path, block)




# import os
# import uuid
# import time
# import numpy as np
# import cv2

# def stackImages(imgArray, scale, labels=[]):
#     rows = len(imgArray)
#     cols = len(imgArray[0])
#     rowsAvailable = isinstance(imgArray[0], list)
#     width = imgArray[0][0].shape[1]
#     height = imgArray[0][0].shape[0]
#     newWidth = int(width * scale)
#     newHeight = int(height * scale)
    
#     if rowsAvailable:
#         for x in range(rows):
#             for y in range(cols):
#                 imgArray[x][y] = cv2.resize(imgArray[x][y], (newWidth, newHeight))
#                 if len(imgArray[x][y].shape) == 2:
#                     imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
#         hor = [np.hstack(imgArray[x]) for x in range(rows)]
#         ver = np.vstack(hor)
#     else:
#         for x in range(rows):
#             imgArray[x] = cv2.resize(imgArray[x], (newWidth, newHeight))
#             if len(imgArray[x].shape) == 2:
#                 imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
#         ver = np.hstack(imgArray)
    
#     return ver

# def rectContour(contours):
#     rectCon = []
#     for i in contours:
#         area = cv2.contourArea(i)
#         if area > 50:
#             peri = cv2.arcLength(i, True)
#             approx = cv2.approxPolyDP(i, 0.02 * peri, True)
#             if len(approx) == 4:
#                 rectCon.append(i)
#     rectCon = sorted(rectCon, key=cv2.contourArea, reverse=True)
#     return rectCon

# def getCornerPoints(cont):
#     peri = cv2.arcLength(cont, True)
#     return cv2.approxPolyDP(cont, 0.02 * peri, True)

# def reorder(points):
#     points = points.reshape((4, 2))
#     newPoints = np.zeros((4, 1, 2), np.int32)
#     add = points.sum(1)
#     newPoints[0] = points[np.argmin(add)]
#     newPoints[3] = points[np.argmax(add)]
#     diff = np.diff(points, axis=1)
#     newPoints[1] = points[np.argmin(diff)]
#     newPoints[2] = points[np.argmax(diff)]
#     return newPoints

# def verticalSplitBoxes(img):
#     cols = np.hsplit(img, 5)
#     boxes = []
#     for c in cols:
#         rows = np.vsplit(c, 10)
#         for box in rows:
#             boxes.append(box)
#     return boxes

# def getAnswerBlocks(img):
#     img = cv2.resize(img, (350, 50))
#     return np.hsplit(img, 7)

# def saveImages(answerBlocks, imageName, boxNumber):
#     cross_path = r"D:\University\fyp\mcq_test_1\cross_Images"
#     empty_path = r"D:\University\fyp\mcq_test_1\empty_Images"
#     os.makedirs(cross_path, exist_ok=True)
#     os.makedirs(empty_path, exist_ok=True)

#     for i, block in enumerate(answerBlocks):
#         gray_block = cv2.cvtColor(block, cv2.COLOR_BGR2GRAY)
#         totalPixels = cv2.countNonZero(gray_block)
#         unique_id = str(uuid.uuid4())[:8]
#         timestamp = int(time.time())
#         filename = f"{imageName}_{boxNumber + i}_{timestamp}_{unique_id}.jpg"

#         if totalPixels < 550:
#             file_path = os.path.join(empty_path, filename)
#         else:
#             file_path = os.path.join(cross_path, filename)
        
#         cv2.imwrite(file_path, block)
