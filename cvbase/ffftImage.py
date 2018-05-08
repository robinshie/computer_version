import numpy as np
import matplotlib.pyplot as plt
import cv2
image = cv2.imread("./imgs/1.jpg")
image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
f = np.fft.fft2(image/255.0)#use 2 dim fft
f_shift = np.fft.fftshift(f)
'''
#Shift the zero-frequency component to the center of the spectrum.
>>> freqs = np.fft.fftfreq(9, d=1./9).reshape(3, 3)
>>> freqs
array([[ 0.,  1.,  2.],
       [ 3.,  4., -4.],
       [-3., -2., -1.]])
>>> np.fft.fftshift(freqs, axes=(1,))
array([[ 2.,  0.,  1.],
       [-4.,  3.,  4.],
       [-1., -3., -2.]])
'''
frequency_tx = 60*np.log(np.abs(f_shift))
f,(a1,a2) = plt.subplots(1,2,figsize=(200,200))
a1.set_title("original image")
a1.imshow(image,cmap='gray')
a2.set_title("requency transform image")
a2.imshow(frequency_tx,cmap='gray')
plt.show()