#coding:utf-8
import matplotlib.pyplot as plt
import numpy as np
import cv2
img = cv2.imread("./imgs/7.jpg")
img_cp = np.copy(img)
img_cp = cv2.cvtColor(img_cp,cv2.COLOR_BGR2HSV)

plt.figure()
f,(a1,a2,a3,a4) = plt.subplots(1,4,figsize=(200,200))
h_img = img_cp[:,:,0]
s_img = img_cp[:,:,1]
v_img = img_cp[:,:,2]
a1.set_title("HSV channel")
a1.imshow(img_cp,cmap='gray')

a2.set_title("H channel")
a2.imshow(h_img,cmap='gray')

a3.set_title("S channel")
a3.imshow(s_img,cmap='gray')

a4.set_title("V channel")
a4.imshow(v_img,cmap='gray')
#plt.imshow(img_cp)
#plt.show()

l_threhold = np.array([100,43,46])#blue range
h_threhold = np.array([124,255,255])

mask = cv2.inRange(img_cp,l_threhold,h_threhold)
masked_img = np.copy(img_cp)
masked_img[mask != 0] = [0,0,0]
plt.figure()
plt.imshow(cv2.cvtColor(masked_img,cv2.COLOR_HSV2RGB))
plt.show()