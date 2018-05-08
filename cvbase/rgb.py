#coding:utf-8
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

rgb_img = mpimg.imread("./imgs/8.jpg");

plt.figure
f,(a1,a2,a3,a4) = plt.subplots(1,4,figsize=(200,200))

r = rgb_img[:,:,0]
g = rgb_img[:,:,1]
b = rgb_img[:,:,2]
a1.set_title('R channel')
a1.imshow(r)
a2.set_title('G channel')
a2.imshow(g)
a3.set_title('B channel')
a3.imshow(b)
a4.set_title("RGB channel")
a4.imshow(rgb_img)
plt.show()

