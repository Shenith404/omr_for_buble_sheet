import numpy as np
import cv2

def stackImages(imgArray, scale, labels=[]):
    """Stack images in a grid with optional labels"""
    rows = len(imgArray)
    cols = len(imgArray[0]) if isinstance(imgArray[0], list) else 1
    
    # Calculate dimensions
    width = imgArray[0][0].shape[1] if isinstance(imgArray[0], list) else imgArray[0].shape[1]
    height = imgArray[0][0].shape[0] if isinstance(imgArray[0], list) else imgArray[0].shape[0]
    
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Prepare images
    if isinstance(imgArray[0], list):
        # 2D array
        for x in range(rows):
            for y in range(cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (new_width, new_height))
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        
        # Create rows
        rows_combined = []
        for x in range(rows):
            rows_combined.append(np.hstack(imgArray[x]))
        
        # Stack vertically
        final_image = np.vstack(rows_combined)
    else:
        # 1D array
        for x in range(rows):
            imgArray[x] = cv2.resize(imgArray[x], (new_width, new_height))
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        
        final_image = np.hstack(imgArray)
    
    # Add labels if provided
    if labels:
        each_img_width = final_image.shape[1] // cols
        each_img_height = final_image.shape[0] // rows
        
        for d in range(rows):
            for c in range(cols):
                text_pos = (c * each_img_width + 10, (d + 1) * each_img_height - 10)
                cv2.putText(final_image, labels[d][c], text_pos, 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
     
    return final_image

def rectContour(contours):
    """Filter and sort rectangular contours by area"""
    rect_con = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 50:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                rect_con.append(cnt)
    
    return sorted(rect_con, key=cv2.contourArea, reverse=True)

def getCornerPoints(cont):
    """Get corner points of a contour"""
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx

def reorder(myPoints):
    """Reorder points to consistent order (tl, tr, bl, br)"""
    myPoints = myPoints.reshape((4, 2))
    myPointsNew = np.zeros((4, 1, 2), np.int32)
    
    add = myPoints.sum(1)
    myPointsNew[0] = myPoints[np.argmin(add)]  # Top-left
    myPointsNew[3] = myPoints[np.argmax(add)]  # Bottom-right
    
    diff = np.diff(myPoints, axis=1)
    myPointsNew[1] = myPoints[np.argmin(diff)]  # Top-right
    myPointsNew[2] = myPoints[np.argmax(diff)]  # Bottom-left
    
    return myPointsNew

def verticalSplitBoxes(img):
    """Split image into answer boxes (5 columns x 10 rows)"""
    cols = np.hsplit(img, 5)
    boxes = []
    
    for col in cols:
        rows = np.vsplit(col, 10)
        for box in rows:
            boxes.append(box)
    
    return boxes

def getAnswerBlocks(img):
    """Split an answer box into 7 parts (for A-G options)"""
    img = cv2.resize(img, (350, 50))
    blocks = np.hsplit(img, 7)
    return blocks

#show answers on the image
def showAnswers(img,answerIndexes,model_answers):
    secW = int(img.shape[1] / 35)
    secH = int(img.shape[0] / 10)
    totalMarks=0

    

    for x in range(0,5):

        for y in range(0, 10):
            myAns = answerIndexes[x * 10 + y]
            correctAns = model_answers[x * 10 + y]+1


            # Center coordinates
            cx = (myAns * secW + 7 * x * secW) + secW // 2
            cy = (y * secH) + secH // 2

            # 1. Draw yellow cross for user answer
            if myAns != -1:
                cv2.line(img, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 255, 255), 2)  # Yellow line
                cv2.line(img, (cx - 10, cy + 10), (cx + 10, cy - 10), (0, 255, 255), 2)  # Yellow line

            # 2. Draw square above cross
            if myAns == correctAns:
                totalMarks += 1
                # Add green rectangle for whole line
                cv2.rectangle(img, (cx - 15, cy - 15), (cx + 15, cy + 15), (0, 255, 0), cv2.FILLED)

    

    return img,totalMarks


#get the none zero answerlength
def getNoneZeroAnswerLength(answers):
    """Get the length of the answers that are not zero"""
    count = 0
    for ans in answers:
        if ans != 0:
            count += 1
    return count
    