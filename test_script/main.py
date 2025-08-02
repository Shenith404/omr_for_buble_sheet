import cv2
import numpy as np
import utils
import os
import model

#constants
path ='../images/test_22.jpeg'
widhtImg = 1025
hightImg = 760
webCamFeed = True

cap = cv2.VideoCapture(0)
cap.set(10,150)

# while True:
# if webCamFeed: success, img = cap.read()
# else: img = cv2.imread(path)

img = cv2.imread(path)
#preprocessing
imgContours=img.copy()
imgBiggestContours=img.copy()
img=cv2.resize(img,(widhtImg,hightImg)) 
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    #convert image to gray
imgBlur=cv2.GaussianBlur(imgGray,(3,3),1)       #apply blur(image_source, kernel_size, sigma)
imgCanny=cv2.Canny(imgBlur,10,50)               #apply canny edge detection (image_source, threshold1, threshold2)

try:
    #finding all contours
    contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #RETR_EXTERNAL External method to find outer edges #CHAIN_APPROX_NONE no need any approximation
    cv2.drawContours(imgContours,contours,-1,(0,255,0),10) # -1 index to draw all contours # (0,255,0) color of contours # 10 thickness of contours
    #find rectangle contours
    rectCon =utils.rectContour(contours)
    biggestContour = utils.getCornerPoints(rectCon[0])
    # secondBiggestContour = utils.getCornerPoints(rectCon[1])
    
    # #print(secoundbiggestContour)
    # if len(rectCon) >= 2:
    #     secondBiggestContour = utils.getCornerPoints(rectCon[1])
    #     text, extractedImg = utils.extract_text_from_box(img, secondBiggestContour)
    #     print("Text from second largest box:", text)
    #     cv2.imshow("Second Largest Box", extractedImg)

    #print(biggestContour)
    if biggestContour.size !=0:
        cv2.drawContours(imgBiggestContours,biggestContour,-1,(0,255,0),20)
        biggestContour=utils.reorder(biggestContour)
        
        pt1=np.float32(biggestContour)
        pt2=np.float32([[0,0],[widhtImg,0],[0,hightImg],[widhtImg,hightImg]])
        matrix=cv2.getPerspectiveTransform(pt1,pt2)
        imgWarpColored=cv2.warpPerspective(img,matrix,(widhtImg,hightImg))
        
        #Apply threshold
        imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)
        imgWarpGray =cv2.convertScaleAbs(imgWarpGray, alpha=1, beta=50)

        thresh = cv2.adaptiveThreshold(imgWarpGray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV, 11, 2)

        # Morphological Opening (remove small white dots)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_open, iterations=1)
        erosion_image =cv2.erode(thresh, kernel_open, iterations=1)


        #get answers boxes
        boxes =utils.verticalSplitBoxes(erosion_image)
        #thresh_boxes =utils.verticalSplitBoxes(erosion_image)




        ############################ GET THE ANSWERS ############################
        answerIndexes=[]
        answerWithModels =[]

        for i in range(len(boxes)):
            answerBoxes = utils.getAnswerBlocks(boxes[i])
            #thresh_answer_blocks =utils.getAnswerBlocks(thresh_boxes[i])[2:6]

         

            # Ignore indices 0, 1, and 6
            answerBoxes = answerBoxes[2:6]
            
           # print("vadf",cv2.countNonZero(thresh_answer_blocks[0]),i)
         
               
            #get answer labels
            answerLabels=[]
            for j in range(len(answerBoxes)):
                #check if the bubble is crossed using the model
                if cv2.countNonZero(answerBoxes[j]) >         0:
                    label, confidence = model.classify_bubble(answerBoxes[j])
                    answerWithModels.append({"box_index": i+1, "answer": j+2, "label": label})
                    if label=="cross_sheets_adpthresh":
                        answerLabels.append(j+2)
                        
            #if answerlabels have only one crossed bubble then add the index of the crossed bubble
            if len(answerLabels)==1:
                answerIndexes.append(answerLabels[0])
            else:
                answerIndexes.append(-1)
                


        print(answerWithModels)  # Debug output

        


        ############################ DISPLAY ANSWERS ############################

        imageResult = utils.showAnswers(imgWarpColored,answerIndexes)
        imageRowDrawing = np.zeros_like(imgWarpColored)
        imageRowDrawing = utils.showAnswers(imageRowDrawing,answerIndexes)
        
        # Fix the perspective transform by using the correct matrix
        inverseMatrix = cv2.getPerspectiveTransform(pt2,pt1)
        inverseImage = cv2.warpPerspective(imageRowDrawing,inverseMatrix,(widhtImg,hightImg))

        # Ensure both images have the same size before blending
        imgFinal = cv2.resize(img.copy(),(widhtImg,hightImg))
        imgFinal = cv2.addWeighted(imgFinal,1,inverseImage,1,0)


        ############################ SAVE THE IMAGE BLOCKS ############################
        # imageName='test_12'
        # for i in range(len(boxes)):

        #     utils.saveImages(utils.getAnswerBlocks(boxes[i]),imageName,i)


        imgBlank = np.zeros_like(img)

        imageArray=([imgContours,imgBiggestContours,imgWarpColored,erosion_image] ,
            [imageResult,imageRowDrawing,imgFinal,imgBlank])
except:
    imgBlank = np.zeros_like(img)

    imageArray=([img,imgBlank,imgBlank,imgBlank] ,
            [imgBlank,imgBlank,imgBlank,imgBlank])




imageStacked =utils.stackImages(imageArray,0.5)

#display image
cv2.imshow('Stack Images', imageStacked)
cv2.waitKey(0)
# if(cv2.waitKey(1) & 0xFF == ord('s')):
#     cv2.imwrite('savedImage.jpg',imageStacked)
#     cv2.waitKey(300)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     exit()
#     break