from cgi import test
import numpy as np
import csv
import sklearn.model_selection as sm
import torch
import pandas as pd
import os

# tr_path = '../machineLearning/base/pytorch/hw1/covid.train.csv'  # path to training data

# data = pd.read_csv(tr_path)
# # print(data)
# data_y = data[:,[40, 41, 42, 43, 57, 58, 59, 60, 61, 75, 76, 77, 78, 79, 93]]
# print(data_y)


# def tet(myName):
#     if myName == 'luan':
#         master = True
#     else :
#         master = False
#     print('jieguoshi :  ',master)

# tet('luan')

x = np.arange(0, 28)
# print(X_index)
result = x.reshape(7,4)
# print(reult)
# print(reult.shape[0])

split_index = np.arange(0,7)
train_index, valid_index = sm.train_test_split(split_index, test_size=0.2, random_state=42)
print(result[train_index])
print(result[valid_index])