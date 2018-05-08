import imgHelp as help
import matplotlib.pyplot as plt
import numpy as np
'''
# 平均亮度
'''
def avg_brightness(img):
    v_img = img[:,:,2]
    sum_brightness = np.sum(v_img)
    return sum_brightness/(v_img.shape[0]*v_img.shape[1])
'''
# 评估标签
'''
def estimate_label(img):
    avg = avg_brightness(img)
    threhold = 110 # 设置明暗范围
    predict_label = 1
    if(avg<threhold):
        predict_label = 0
    return predict_label
'''
# 收集错误分类
'''
def misclassified_images(test_imges):
    misclassified_images = []
    for img in test_imges:
        current_img = img[0]
        true_lab = img[1]
        pre_lab = estimate_label(current_img)
        if(true_lab!=pre_lab):
            misclassified_images.append((current_img,true_lab,pre_lab))
    return misclassified_images
img_train_dataSet = help.load_dataSet("./day","HSV") + help.load_dataSet("./night","HSV")
img_test_dataSet = help.load_dataSet("./test","HSV")
stded_train_img_dataSet = help.standartize(img_train_dataSet)
stded_test_img_dataSet = help.standartize(img_test_dataSet)

misclassified_images=misclassified_images(stded_test_img_dataSet)

print("准确率:",1-(len(misclassified_images)/len(img_test_dataSet)))
#print(estimate_label(stded_test_img_dataSet[1][0]))
#print(stded_img_dataSet[0][1])
plt.imshow(misclassified_images[0][0])
#print(stded_test_img_dataSet[1][1])
#print(misclassified_images[1][2])
#print(estimate_label(stded_img_dataSet[11][0]))
plt.show()