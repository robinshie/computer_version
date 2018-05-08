import cv2 
import matplotlib.pyplot as plt  
plt.rcParams['figure.figsize'] = [14.0, 7.0] 
image1 = cv2.imread('./imgs/m.jpg') 
image2 = cv2.imread('./imgs/m.jpg')
image2 = cv2.resize(image2,(100,100))
training_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB) 
query_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
training_gray = cv2.cvtColor(training_image, cv2.COLOR_BGR2GRAY) 
query_gray = cv2.cvtColor(query_image, cv2.COLOR_BGR2GRAY)


orb = cv2.ORB_create(1000,2.0)

kp_train_gray,descriptions_train = orb.detectAndCompute(training_gray,None)
kp_quire_gray,descriptions_quire = orb.detectAndCompute(query_gray,None)

bf = cv2.BFMatcher(cv2.NORM_L2SQR,crossCheck = True)

matches = bf.match(descriptions_train,descriptions_quire)

matches = sorted(matches,key=lambda x:x.distance)

result = cv2.drawMatches(training_gray,kp_train_gray,query_gray,kp_quire_gray,matches[:200],query_gray,flags=2)

f,(a1,a2,a3) = plt.subplots(1,3)
a1.imshow(training_gray)
a1.set_xlim(0,image1.shape[1])
a1.set_ylim(0,image1.shape[0])
a2.imshow(query_gray)
a2.set_xlim(0,image1.shape[1])
a2.set_ylim(0,image1.shape[0])
a3.imshow(result)
plt.show()