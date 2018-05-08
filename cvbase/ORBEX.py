import cv2 

import matplotlib.pyplot as plt

import copy

plt.rcParams['figure.figsize'] = [14.0, 7.0] 

image1 = cv2.imread('./imgs/m.jpg') 

image2 = cv2.imread('./imgs/basa.jpg') 

training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) 

query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) 

plt.rcParams['figure.figsize'] = [34.0, 34.0] 

training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY) 

query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(5000, 2.0)

keypoints_train, descriptors_train = orb.detectAndCompute(training_gray, None) 

keypoints_query, descriptors_query = orb.detectAndCompute(query_gray, None)

query_img_keyp = copy.copy(query_image)

cv2.drawKeypoints(query_image, keypoints_query, query_img_keyp, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

plt.rcParams['figure.figsize'] = [34.0, 34.0]
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
matches = bf.match(descriptors_train, descriptors_query)
matches = sorted(matches, key = lambda x : x.distance)
resultImg = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matches[:85], query_gray, flags = 2)
plt.title('Best Matching Points', fontsize = 30) 
plt.imshow(resultImg) 
plt.show()
