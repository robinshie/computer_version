import cv2 
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize'] = [20,10] 
image = cv2.imread('./imgs/m.jpg') 
training_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
training_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
'''
plt.subplot(121) 
plt.title('Original Training Image') 
plt.imshow(training_image) 
plt.subplot(122) 
plt.title('Gray Scale Training Image') 
plt.imshow(training_gray, cmap = 'gray') 
plt.show() 
'''
import copy

plt.rcParams['figure.figsize'] = [14,7]

orb = cv2.ORB_create(100,2.0)

keypoints,descripor = orb.detectAndCompute(training_gray,None)


keyp_without_size = copy.copy(training_image)
keyp_with_size = copy.copy(training_image)

cv2.drawKeypoints(training_image,keypoints,keyp_without_size,color = (0, 255, 0))

cv2.drawKeypoints(training_image,keypoints,keyp_with_size,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.subplot(121)
plt.title("Keypoints without size or orientation")
plt.imshow(keyp_without_size)
plt.subplot(122)
plt.title("Keypoints without size or orientation")
plt.imshow(keyp_with_size)

print("\n Number of kepoints detected",len(keypoints))
plt.show()