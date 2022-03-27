from turtle import shape
import numpy as np
import scipy.misc as sm
import scipy.ndimage as sn
import sklearn.cluster as sc
import matplotlib.pyplot as mp
import cv2 as cv

def quant(image, n_clusters):
    x = image.reshape(-1,1)
    model = sc.KMeans(n_clusters=n_clusters)
    model.fit(x)
    y = model.labels_
    centers = model.cluster_centers_.ravel()
    return centers[y].reshape(image.shape)


image = cv.imread('../machineLearning/data/test.jpeg')
print(image.shape) 
quant1 = quant(image, 4)
print(quant1.shape)
# quant2 = quant(image, 8)
# quant3 = quant(image, 12)
# mp.figure('images quant', facecolor='lightgray')
# mp.subplot(221)
# mp.title("nornaml", fontsize=16)
# mp.axis('off')
# mp.imshow(image,cmap='gray')
# mp.subplot(222)
# mp.title("KMeans-4", fontsize=16)
# mp.axis('off')
# mp.imshow(quant1,cmap='gray')
# mp.subplot(223)
# mp.title("KMeans-8", fontsize=16)
# mp.axis('off')
# mp.imshow(quant2,cmap='gray')
# mp.subplot(224)
# mp.title("KMeans-10", fontsize=16)
# mp.axis('off')
# mp.imshow(quant3,cmap='gray')
# mp.tight_layout
# mp.show()