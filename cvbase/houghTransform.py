import numpy as np
import matplotlib.pyplot as plt
import cv2
import imgHelp
image = cv2.imread('imgs/5.jpg')

gray =cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_RGB2GRAY),(7,7),3)

fft_img = imgHelp.fftPic(gray,30)

h_threshold = 100
l_threshold = 50


edges = cv2.Canny(gray,l_threshold,h_threshold)

rho = 1
theta = np.pi/180 
threshold = 150 
min_line_length = 1
max_line_gap = 10
line_image = np.copy(image)
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),min_line_length, max_line_gap) 
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

circles_im = np.copy(image)
circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 1,minDist=35,param1=100,param2=20, minRadius=20,maxRadius=40) 
circles = np.uint32(np.around(circles)) 
for i in circles[0,:]: 
    cv2.circle(circles_im,(i[0],i[1]),i[2],(0,255,0),2) 
f,(a1,a2,a3,a4,a5,a6) = plt.subplots(1,6,figsize=(600,400))

a1.set_title("oraginal image")
a1.imshow(image)

a2.set_title("gra")
a2.imshow(gray,cmap="gray")

a3.set_title("fft")
a3.imshow(fft_img,cmap="gray")

a4.set_title("edges")
a4.imshow(edges,cmap="gray")

a5.set_title("line_image")
a5.imshow(line_image)

a6.set_title("circles_im")
a6.imshow(circles_im)
plt.show()