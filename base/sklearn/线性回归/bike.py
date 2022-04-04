import numpy as np
import sklearn.utils as su
import sklearn.ensemble as se
import sklearn.metrics as sm


# data = []
# # with open('../machineLearning/data/bike_dat.csv','r')as f:
# with open('../machineLearning/data/bike_day.csv','r')as f:
#   ' for line in f.readlines():
#         data.append(line[:-1].split(','))
data = np.loadtxt('../machineLearning/data/bike_day.csv' ,dtype='U20', delimiter=',')
print(data[:2  ])

# header = data[0][2:13]
# x = np.array(data[1:])[:,2:13].astype('f8')
# y = np.array(data[1:])[:,-1].astype('f8')

# x,y = su.shuffle(x,y,random_state=8)
# train_size = int(len(x)*0.85)
# train_x,train_y,test_x,test_y = x[:train_size], y[:train_size],x[train_size:],y[train_size:]
# model = se.RandomForestRegressor(max_depth=8,n_estimators=1000)
# model.fit(train_x,train_y)
# pred_y = model.predict(test_x)
# print(sm.r2_score(test_y,pred_y))