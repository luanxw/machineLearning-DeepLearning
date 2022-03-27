# -*- coding=utf-8 -*-
import sklearn.linear_model as lm
import pickle 
import numpy as np


tran_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
tran_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])
tran_x = tran_x.reshape(-1,1)
 
model = lm.LinearRegression()
model.fit(tran_x,tran_y)

# with open('../machineLearning/model/linear.luan', 'wb') as f:
#     pickle.dump(model, f)
#     print('save success!')

with open('../machineLearning/model/linear.luan', 'r') as f:
    model = pickle.load(f)
    print('load success!')
