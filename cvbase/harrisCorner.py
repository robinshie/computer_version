import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('./imgs/10.jpg')
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
dst = cv2.cornerHarris(gray,2,3,0.04)
pointdst = cv2.dilate(dst,None)

threshold = 0.1*dst.max()

for i in range(0,dst.shape[0]):
    for j in range(0,dst.shape[1]):
        if(dst[i,j]>threshold):
            cv2.circle(img,(j,i),1,(255,0,0),2)
plt.imshow(img)
plt.show()