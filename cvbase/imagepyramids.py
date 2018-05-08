import numpy as np 
import matplotlib.pyplot as plt 
import cv2 
image = cv2.imread('imgs/4.jpg') 
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

level_1 = cv2.pyrDown(image)
level_2 = cv2.pyrDown(level_1)
level_3 = cv2.pyrDown(level_2)

f,(a1,a2,a3,a4) = plt.subplots(1,4,figsize=(20,10))
a1.set_title('oraginal')
a1.imshow(image)

a2.imshow(level_1)
a2.set_xlim(0,image.shape[1])
a2.set_ylim(0,image.shape[0])

a3.imshow(level_2)
a3.set_xlim(0,image.shape[1])
a3.set_ylim(0,image.shape[0])

a4.imshow(level_3)
a4.set_xlim(0,image.shape[1])
a4.set_ylim(0,image.shape[0])

plt.show()