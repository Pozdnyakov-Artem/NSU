import os
import cv2
import numpy as np

images = sorted(os.listdir("images"))
labels = sorted(os.listdir("labels"))

mas=list(zip(images,labels))

task = 2

for image,label in mas:

    img = cv2.resize(cv2.imread("images\\"+image),(500,500))
    if task == 1:
        lbl = cv2.imread("labels\\"+label)
        # print(lbl.shape)
    else:
        lbl = cv2.imread("labels\\"+label,cv2.IMREAD_GRAYSCALE)

    if img.shape != lbl.shape:
        lbl = cv2.resize(lbl, (img.shape[0], img.shape[1]))

    if task == 1:
        comb = np.hstack([img,lbl])
        cv2.imshow("win",comb)
        cv2.waitKey(0)
    else:
        counturs, hier = cv2.findContours(lbl,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img,counturs,-1,(0,255,0),3)
        cv2.imshow("win",img)
        cv2.waitKey(0)