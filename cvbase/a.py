import numpy as np  
import cv2  
import matplotlib.pyplot as plt
img=cv2.imread("./imgs/1.jpg")  
#cv2.imshow("temp",img)  

img90=np.rot90(img)  
  
plt.imshow(img90)  
plt.show()