import os
from statistics import mode

from matplotlib.pyplot import axis
import cv2 as cv
import numpy as np
import sklearn.preprocessing as sp
import sklearn.svm as svm 
import sklearn.metrics as sm



train_x , train_y = [],[]
def search_flies(directory):
    '''
    检索 directory 目录下所有的jpg文件返回字典目录
    '''
    objects = {}
    for curdir ,subdir , files in  os.walk(directory):
        for file in files:
            if file.endswith('jpg'):
                label = curdir.split(os.path.sep)[-1]
                if label not in objects:
                    objects[label] = []
                url = os.path.join(curdir, file)
                objects[label].append(url)
    return objects
train_urls = search_flies('../machineLearning/data/train')

for label , files in train_urls.items():
    for file in files:
        # 提取图片的特征矩阵、整理样本
        image = cv.imread(file)
        #拿到灰度图
        gray =  cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        #拿到高度和宽度
        h , w = gray.shape
        #要把较小的变变成200、大的那一边自适应, 防止失帧
        f = 200 / min(h,w)
        #按照比例缩放图片
        cv.resize(gray, None, fx=f,fy=f)
        # 创建STAR特征点检测器
        shift = cv.SIFT_create()
        # 检测出gray图像所有的特征点
        keypoints = shift.detect(gray)
        # desc:(n, 128)的特征值矩阵
        _, desc = shift.compute(gray, keypoints)
        #垂直方向求均值
        sample = np.mean(desc, axis=0)
        train_x.append(sample)
        train_y.append(label)
train_x = np.array(train_x)
encoder = sp.LabelEncoder()
train_y_labeled = encoder.fit_transform(train_y)
model = svm.SVC(kernel='poly',degree=3, probability=True)
model.fit(train_x,train_y_labeled)
predict_y = model.predict(train_x) 
model.fit(train_x, train_y)
print('------------分类报告---------------')
print(sm.classification_report(train_y_labeled,predict_y))
proba = model.predict_proba(train_x) 
print('------------置信概率---------------')
print(proba)
print('------------预测结果---------------')
for label ,prod in zip(encoder.inverse_transform(predict_y),proba.max(axis=1)):
    print(label,prod)

