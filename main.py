import cv2
import numpy as np
import utils
import os
import model

#constants
path ='test_11.jpeg'
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
imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)
imgCanny=cv2.Canny(imgBlur,10,50)

try:
    #finding all contours
    contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgContours,contours,-1,(0,255,0),10)
    #find rectangle contours
    rectCon =utils.rectContour(contours)
    biggestContour = utils.getCornerPoints(rectCon[0])

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
        imgThresh=cv2.threshold(imgWarpGray,170,255,cv2.THRESH_BINARY_INV)[1]

        #get answers boxes
        boxes =utils.verticalSplitBoxes(imgThresh)



        ############################ GET THE ANSWERS ############################
        answerIndexes=[]
        answerWithModels =[]

        for i in range(len(boxes)):
            answerBoxes = utils.getAnswerBlocks(boxes[i])

            # Ignore indices 0, 1, and 6
            answerBoxes = answerBoxes[2:6]
            #get answer labels
            answerLabels=[]
            for j in range(len(answerBoxes)):
                #check if the bubble is crossed using the model
                if cv2.countNonZero(answerBoxes[j]) > 0:
                    label, confidence = model.classify_bubble(answerBoxes[j])
                    answerWithModels.append({"box_index": i+1, "answer": j+2, "label": label})
                    if label=="Crossed_Bubble" or label=="Cross_Removed_Bubble":
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

        imageArray=([imgContours,imgBiggestContours,imgWarpColored,imgThresh] ,
            [imageResult,imageRowDrawing,imgFinal,imgBlank])
except:
    imgBlank = np.zeros_like(img)

    imageArray=([img,imgBlank,imgBlank,imgBlank] ,
            [imgBlank,imgBlank,imgBlank,imgBlank])




imageStacked =utils.stackImages(imageArray,0.3)

#display image
cv2.imshow('Stack Images', imageStacked)
cv2.waitKey(0)
# if(cv2.waitKey(1) & 0xFF == ord('s')):
#     cv2.imwrite('savedImage.jpg',imageStacked)
#     cv2.waitKey(300)
# if cv2.waitKey(1) & 0xFF == ord('q'):
#     exit()
#     break