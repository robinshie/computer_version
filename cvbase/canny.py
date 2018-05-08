import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import numpy as np

image = cv2.imread("./imgs/2.jpg")

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

wide = cv2.Canny(gray,30,100)
tight = cv2.Canny(gray,180,240)

f,(a1,a2) = plt.subplots(1,2,figsize=(200,200))
a1.set_title("wide")
a1.imshow(wide,cmap='gray')
a2.set_title("tight")
a2.imshow(tight,cmap='gray')
plt.show()