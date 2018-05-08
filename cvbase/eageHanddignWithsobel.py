import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import numpy as np

image = mpimg.imread("./imgs/1.jpg")

gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

core_sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
core_sobely = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
core_self= np.array([[-1,-2,-1,1,3],[-2,-3,-2,3,4],[0,  0, 0,0,0],[-2,-3,-2,3,4],[-1,-2,-1,1,3]])

img_sobelx = cv2.filter2D(gray,-1,core_sobelx)
img_sobely = cv2.filter2D(gray,-1,core_sobely)
img_coreself = cv2.filter2D(gray,-1,core_self)

f,(a1,a2,a3,a4) = plt.subplots(1,4,figsize=(200,200))
a1.set_title("oraginal image")
a1.imshow(gray,cmap="gray")

a2.set_title("sobel x")
a2.imshow(img_sobelx,cmap="gray")

a3.set_title("sobel y")
a3.imshow(img_sobely,cmap="gray")

a4.set_title("self core")
a4.imshow(img_coreself,cmap="gray")

plt.show()
