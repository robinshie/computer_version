import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import cv2
import numpy as np

image = cv2.imread("./imgs/9.jpg")

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

gray_blur = cv2.GaussianBlur(gray,(9,9),5)

f = np.fft.fft2(gray/225.0)
f_shift = np.fft.fftshift(f)
grayfft = 10 * np.log(np.abs(f_shift))

core_sobelx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
core_sobely = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])


gray_blur_sobelx = cv2.filter2D(gray_blur,2,core_sobelx)
gray_blur_sobely = cv2.filter2D(gray_blur,2,core_sobely)
gray_blur =  cv2.filter2D(gray,2,core_sobelx)

f,(a0,a1,a2,a3,a4,a5) = plt.subplots(1,6,figsize=(30,30))

a5.set_title("fft gray")
a5.imshow(grayfft,cmap='gray')

a0.set_title("oraginal gray")
a0.imshow(gray,cmap='gray')

a1.set_title("oraginal gray filter")
a1.imshow(gray_blur,cmap='gray')

a2.set_title("blur gray")
a2.imshow(gray_blur,cmap='gray')

a3.set_title("blur gray with sobel x")
a3.imshow(gray_blur_sobelx,cmap='gray')

a4.set_title("blur gray with sobel y")
a4.imshow(gray_blur_sobely,cmap='gray')

retval,binary_image = cv2.threshold(gray_blur_sobelx,100,225,cv2.THRESH_BINARY)

plt.figure()
plt.imshow(binary_image,cmap='gray')
plt.show()

