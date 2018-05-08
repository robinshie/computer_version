import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread('./imgs/11.jpg',0)
kernel = np.ones((5,5),np.uint8)
dilate_img = cv2.dilate(img,kernel,iterations=1)
erosion_img = cv2.erode(img,kernel,iterations=1)

opening_img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
closing_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


f,(a1,a2,a3,a4,a5) = plt.subplots(1,5,figsize=(600,400))

a1.set_title("oraginal image")
a1.imshow(img,cmap="gray")

a2.set_title("dilate_img")
a2.imshow(dilate_img,cmap="gray")

a3.set_title("erosion_img")
a3.imshow(erosion_img,cmap="gray")

a4.set_title("opening_img")
a4.imshow(opening_img,cmap="gray")

a5.set_title("closing_img")
a5.imshow(closing_img,cmap="gray")
plt.show()