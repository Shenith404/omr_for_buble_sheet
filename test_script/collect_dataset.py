import cv2
import numpy as np
import utils
import os

#constants
images_dir = os.path.join(r"D:\empty_sheets")
image_paths=[]
if os.path.exists(images_dir):
    image_paths.extend([
        os.path.join(images_dir, f) 
        for f in sorted(os.listdir(images_dir))
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
    ])

for path in image_paths:
    widhtImg = 1025
    hightImg = 760




    img = cv2.imread(path)
    #preprocessing
    imgContours=img.copy()
    imgBiggestContours=img.copy()
    img=cv2.resize(img,(widhtImg,hightImg)) 
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    #convert image to gray
    imgBlur=cv2.GaussianBlur(imgGray,(5,5),1)       #apply blur(image_source, kernel_size, sigma)
    imgCanny=cv2.Canny(imgBlur,10,50)               #apply canny edge detection (image_source, threshold1, threshold2)

    try:
        #finding all contours
        contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #RETR_EXTERNAL External method to find outer edges #CHAIN_APPROX_NONE no need any approximation
        cv2.drawContours(imgContours,contours,-1,(0,255,0),10) # -1 index to draw all contours # (0,255,0) color of contours # 10 thickness of contours
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
            boxes =utils.verticalSplitBoxes(imgWarpGray)


            ############################ SAVE THE IMAGE BLOCKS ############################
            
            for i in range(len(boxes)):

                utils.saveImages(utils.getAnswerBlocks(boxes[i])[2:6],i)



            
    except:
        imgBlank = np.zeros_like(img)

        




