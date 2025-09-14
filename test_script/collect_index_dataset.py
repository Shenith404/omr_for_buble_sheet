import time
import uuid
import cv2
import numpy as np
import utils
import os

#constants
try:
    images_dir = os.path.join(r"C:\Users\Shenith Bandara\Downloads\Text _mages\images")
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
        imgBlur=cv2.GaussianBlur(imgGray,(3,3),1)       #apply blur(image_source, kernel_size, sigma)
        imgCanny=cv2.Canny(imgBlur,10,50)               #apply canny edge detection (image_source, threshold1, threshold2)

        try:
            #finding all contours
            contours, hierarchy = cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #RETR_EXTERNAL External method to find outer edges #CHAIN_APPROX_NONE no need any approximation
            cv2.drawContours(imgContours,contours,-1,(0,255,0),10) # -1 index to draw all contours # (0,255,0) color of contours # 10 thickness of contours
            #find rectangle contours
            rectCon =utils.rectContour(contours)
            indexContour = utils.getCornerPoints(rectCon[1])
            cv2.drawContours(imgBiggestContours,indexContour,-1,(0,255,0),20)
            indexContour=utils.reorder(indexContour)
            
            pt1=np.float32(indexContour)
            pt2=np.float32([[0,0],[1000,0],[0,100],[1000,100]])
            matrix=cv2.getPerspectiveTransform(pt1,pt2)
            imgWarpColored=cv2.warpPerspective(img,matrix,(1000,100))

            imgWarpGray=cv2.cvtColor(imgWarpColored,cv2.COLOR_BGR2GRAY)

            #save index contour image
            save_path_cross_Images = r"C:\Users\Shenith Bandara\Downloads\Text _mages\index_images"
            os.makedirs(save_path_cross_Images, exist_ok=True)
            unique_id = str(uuid.uuid4())[:8]  # First 8 characters of UUID
            timestamp = int(time.time())  # Current timestamp
            filename = f"{timestamp}_{unique_id}.jpg"

            #if totalPixels < 550:  # Empty image condition
            file_path = os.path.join(save_path_cross_Images, filename)
            # else:  # Marked image condition
            #     file_path = os.path.join(save_path_cross_Images, filename)

            cv2.imwrite(file_path, imgWarpGray)

        

           


                
        except:
            imgBlank = np.zeros_like(img)
except Exception as e:
    print(e,"Error processing images. Please check the image directory and file formats.")
        




