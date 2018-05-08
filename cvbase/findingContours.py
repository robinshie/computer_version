import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import numpy as np

image = cv2.imread("./imgs/9.jpg")
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
canny_img = cv2.Canny(gray,255,255)
retval, contours, hierarchy  = cv2.findContours(canny_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours_image = np.copy(image)
contours_image = cv2.drawContours(contours_image, contours, -1, (0,255,0), 3)
print(len(contours))
plt.imshow(contours_image)
plt.show()