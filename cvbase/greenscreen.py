import matplotlib.pyplot as plt
import numpy as np
import cv2

img = cv2.imread("./imgs/8.jpg")

img_cp = np.copy(img)

img_cp = cv2.cvtColor(img_cp,cv2.COLOR_BGR2RGB)

l_threhold = np.array([0,180,0])
h_threhold = np.array([255,255,255])

mask = cv2.inRange(img_cp,l_threhold,h_threhold)

masked_img = np.copy(img_cp)

masked_img[mask!=0] = [0,0,0]

img = cv2.imread("./imgs/4.jpg")
img = cv2.resize(img,(img_cp.shape[1],img_cp.shape[0]))

img_back = np.copy(img)

img_back = cv2.cvtColor(img_back,cv2.COLOR_BGR2RGB)

img_back_maked = np.copy(img_back)
img_back_maked[mask==0] = [0,0,0]

plt.imshow(img_back_maked+masked_img)
plt.show()