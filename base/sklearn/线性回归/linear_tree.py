# -*- coding: utf-8 -*-

import sklearn.datasets as sd
import sklearn.tree as st
import sklearn.utils as su
import sklearn.metrics as sm
import sklearn.ensemble as se
# 决策树


boston = sd.load_boston()
# print(boston.data)  # 输入集
# print(boston.target) # 输出集
print(boston.feature_names) # 输入数据的特征名
x,y = su.shuffle(boston.data, boston.target, random_state=8)
train_size = int(len(x)*0.8)
train_x,train_y,Test_x,Test_y = x[:train_size],y[:train_size],x[train_size:],y[train_size:]

model = st.DecisionTreeRegressor(max_depth=6)
model.fit(train_x,train_y)
pred_test_y = model.predict(Test_x)
# print(pred_test_y)
print('第一次训练',sm.r2_score(Test_y,pred_test_y))
#构建200可决策树进行训练
se.AdaBoostClassifier(model,n_estimators=400,random_state=7)
model.fit(train_x,train_y)
pred_y = model.predict(Test_x)
print('第二次训练',sm.r2_score(Test_y,pred_y))