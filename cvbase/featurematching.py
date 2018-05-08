import cv2 
import matplotlib.pyplot as plt  
plt.rcParams['figure.figsize'] = [14.0, 7.0] 
image1 = cv2.imread('./imgs/m.jpg') 
image2 = cv2.imread('./imgs/m.jpg') 
training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) 
query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB) 
training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY) 
query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(100,2.0)

keypoints_train,descriptors_train = orb.detectAndCompute(training_gray,None)
keypoints_query,descriptors_query = orb.detectAndCompute(query_gray,None)

bf = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matchs = bf.match(descriptors_train,descriptors_query)

matchs=sorted(matchs,key=lambda x : x.distance)

result = cv2.drawMatches(training_gray, keypoints_train, query_gray, keypoints_query, matchs[:300], query_gray, flags = 2)

plt.title('Best Matching Points') 

plt.imshow(result) 

plt.show() 