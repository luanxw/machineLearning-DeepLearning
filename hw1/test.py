from cgi import test
import numpy as np
import csv
import torch
import pandas as pd
import os

# tr_path = '../machineLearning/base/pytorch/hw1/covid.train.csv'  # path to training data

# data = pd.read_csv(tr_path)
# # print(data)
# data_y = data[:,[40, 41, 42, 43, 57, 58, 59, 60, 61, 75, 76, 77, 78, 79, 93]]
# print(data_y)


def tet(myName):
    if myName == 'luan':
        master = True
    else :
        master = False
    print('jieguoshi :  ',master)

tet('luan')