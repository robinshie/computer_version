import cv2
import os
import numpy as np
'''
resize 图片
'''
def standartize_input(img):
    return cv2.resize(img,(244,224))
'''
编辑标签
'''
def encode(label):
    numerical_val =0
    if label == 'day':
        numerical_val = 1
    return numerical_val
'''
归一化图片
'''
def standartize(imglist):
    standard_list = []
    for item in imglist:
        img = item[0]
        lab = item[1]
        std_img = standartize_input(img)
        binary_lab = encode(lab)
        standard_list.append((std_img,binary_lab))
    return standard_list
def load_dataSet(dir,colortype="RGB"):
    img_dateSet = []
    for i in os.listdir(dir):
        path = dir +"/"+i
        lable = "night"
        img = cv2.imread(path)
        if(colortype == 'RGB'):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        elif(colortype == 'HSV'):
            img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)            
        if path.__contains__("day"):
            lable = "day"
        img_dateSet.append((img,lable))
    return img_dateSet  
def fftPic(img,num):
    f = np.fft.fft2(img/255.0)#use 2 dim fft
    f_shift = np.fft.fftshift(f)
    return num*np.log(np.abs(f_shift))