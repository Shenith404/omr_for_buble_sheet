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

#get the none zero answerlength
# def getNoneZeroAnswerLength(answers):
#     """Get the length of the answers that are not zero"""
#     count = 0
#     try :
#         for i in len(answers):
#             if answers[i] != 0:
#                 count += 1
#     except Exception as e:
#         print(f"Error in getNoneZeroAnswerLength: {e}")
#         return 50

#     return count

#check the if any answer is zero in mid of the answers
# def checkZeroInMid(answers):
#     """Check if there is a zero in the middle of the answers"""
#     totalQuestions = getNoneZeroAnswerLength(answers)
#     for i,ans in enumerate(answers):
#         if  (i > totalQuestions - 1):
#             return True
#         elif (ans == 0) :
#             return f"answer {i+1} cannot be empty"
#         elif (ans>4) or (ans < 0):
#             return f"answer {i+1} is not valid, it should be between 0 and 4"
    
#     return True

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
            correct_cx = (correctAns * secW + 7 * x * secW) + secW // 2
            if myAns == correctAns:
                # Green square for correct answer
                totalMarks+=1
                cv2.rectangle(img, (cx - 10, cy - 10), (cx + 10, cy + 10), (0, 255, 0), cv2.FILLED)
            else:
                # Red square for incorrect answer
                cv2.rectangle(img, (correct_cx - 10, cy - 10), (correct_cx + 10, cy + 10), (0, 0, 255), cv2.FILLED)


    return img,totalMarks

def process_omr_sheet_without_model( image, detected_answers, model_answers):
    """
    Process a single OMR sheet with optimized operations
    Returns:
    - answers: List of detected answers (1-based index)
    - marked_image: Image with marked answers
    """
    # Step 1: Preprocessing with optimized operations
    img = cv2.resize(image, (1025, 760))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 10, 50)

    

    # Step 2: Contour Detection with area filtering
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    rects = rectContour(contours)
    
    if not rects:
        return  img

    # Step 3: Perspective Transform with error checking
    biggest = getCornerPoints(rects[0])
    if biggest.size == 0:
        return  img
        
    biggest = reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [1025, 0], [0, 760], [1025, 760]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(img, matrix, (1025, 760))

    # Step 4: Adaptive Thresholding for better robustness
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


    # Step 6: Generate Results with optimized drawing
    drawing = np.zeros_like(warped)
    try:
        drawing,total_marks = showAnswers(drawing, detected_answers, model_answers)
    except Exception as e:
        print(f"Error in showAnswers: {e}")
        return img
    inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
    inv_drawing = cv2.warpPerspective(drawing, inv_matrix, (img.shape[1], img.shape[0]))
    final_img = cv2.addWeighted(img, 1, inv_drawing, 1, 0)
    cv2.putText(final_img, f"Total Marks: {total_marks}/50", (50, 700), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 250), 2)


    return  final_img
