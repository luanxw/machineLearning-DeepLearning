import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras import mdoels,layers,losses,metrics,callbacks

path = '../machineLearning/data/covid/'

tr_name = 'covid.train.csv'  # name of training data
tt_name = 'covid.test.csv'   # name of to testing data

train_data = pd.read_csv(path+tr_name)  # 训练数据
test_data = pd.read_csv(path+tt_name)   # 训练数据

# print(test_data.head(2))
# print(test_data.info())

#由于id列用不到，删除id列
train_data.drop(['id'],axis = 1, inplace = True) 
test_data.drop(['id'], axis= 1,inplace=True)

cols = list(train_data.columns) #拿到特征列名称
# print(clos)

# WI列是states one-hot编码最后一列，取值为0或1，后面特征分析时需要把states特征删掉
WI_index = cols.index('WI')
# print(WI_index)
# print(train_data.iloc[:, 40:].describe()) #从上面可以看出wi 列后面是cli, 所以列索引从40开始， 并查看这些数据分布
# print(test_data.iloc[:, 40:].describe()) #查看测试集数据分布，并和训练集数据分布对比，两者特征之间数据分布差异不是很大

# plt.figure('Training data', facecolor='lightgray')
# plt.title("cli -- tested_positive ")
# # plt.subplot(511)
# plt.grid()
# plt.xlabel('cli')
# plt.ylabel('tested_positive')
# plt.scatter(train_data.loc[:, 'cli'], train_data.loc[:, 'tested_positive.2']) #肉眼分析cli特征与目标之间相关性
# plt.legend()

# plt.title("ili -- tested_positive.2 ")
# # plt.subplot(521)
# plt.grid()
# plt.xlabel('ili')
# plt.ylabel('tested_positive.2')
# plt.scatter(train_data.loc[:, 'ili'], train_data.loc[:, 'tested_positive.2'])
# plt.legend()


# plt.title("cli -- ili ")
# # plt.subplot(531)
# plt.grid()
# plt.xlabel('cli')
# plt.ylabel('ili')
# plt.scatter(train_data.loc[:, 'cli'], train_data.loc[:, 'ili'])  #cli 和ili两者差不多，所以这两个特征用一个就行
# plt.legend()



# plt.title("tested_positive -- tested_positive.2 ")
# # plt.subplot(541)
# plt.grid()
# plt.xlabel('tested_positive')
# plt.ylabel('tested_positive.2')
# plt.scatter(train_data.loc[:,  'tested_positive'], train_data.loc[:, 'tested_positive.2']) #day1 目标值与day3目标值相关性，线性相关的
# plt.legend()



# plt.title("tested_positive.1 -- tested_positive.2 ")
# # plt.subplot(551)
# plt.grid()
# plt.xlabel('tested_positive.1')
# plt.ylabel('tested_positive.2')
# plt.scatter(train_data.loc[:,  'tested_positive.1'], train_data.loc[:, 'tested_positive.2']) #day2 目标值与day3目标值相关性，线性相关的
# plt.legend()

# plt.show()

data_corr = train_data.iloc[:, 40:].corr() # print(data_corr) #上面手动分析太累，还是利用corr方法自动分析

with open (path+'covid.train.csv','w') as f:
    for item in data_corr:
        f.writelines(item)
        f.writelines('\r\n')

target_col = data_corr['tested_positive.2']
# print(target_col)
feautures = target_col[target_col > 0.8]
print('------------------------------与预测结果线性相关大于80%的特征------------------------------------')
print(feautures)
feature_cols = feautures.index.tolist()  #将选择特征名称拿出来
# feature_cols.pop() #去掉test_positive标签  pop不填参数,默认最后一个
# print(feature_cols)
# feats_selected = [cols.index(col) for col in feature_cols]  #获取该特征对应列索引编号，后续就可以用feats + feats_selected作为特征值
# print(feats_selected)

# def plot_learning_curve(loss_record, title=''):
#     ''' Plot learning curve of your DNN (train & dev loss) '''
#     total_step = len(loss_record['train'])
#     x_1 = range(total_step)
#     x_2 = x_1[::len(loss_record['train']) // len(loss_record['dev'])]
#     plt.figure(figsize=(6, 4))
#     plt.plot(x_1, loss_record['train'], c='tab:red', label='train')
#     plt.plot(x_2, loss_record['dev'], c='tab:cyan', label='dev')
#     plt.ylim(0.0, 5.)
#     plt.xlabel('Training steps')
#     plt.ylabel('MSE loss')
#     plt.title('Learning curve of {}'.format(title))
#     plt.legend()
#     plt.show()



# def plot_pred(dv_set, model, device, lim=35., preds=None, targets=None):
#     ''' Plot prediction of your DNN '''
#     if preds is None or targets is None:
#         model.eval()
#         preds, targets = [], []
#         for x, y in dv_set:
#             x, y = x.to(device), y.to(device)
#             with torch.no_grad():
#                 pred = model(x)
#                 preds.append(pred.detach().cpu())
#                 targets.append(y.detach().cpu())
#         preds = torch.cat(preds, dim=0).numpy()
#         targets = torch.cat(targets, dim=0).numpy()

#     plt.figure(figsize=(5, 5))
#     plt.scatter(targets, preds, c='r', alpha=0.5)
#     plt.plot([-0.2, lim], [-0.2, lim], c='b')
#     plt.xlim(-0.2, lim)
#     plt.ylim(-0.2, lim)
#     plt.xlabel('ground truth value')
#     plt.ylabel('predicted value')
#     plt.title('Ground Truth v.s. Prediction')
#     plt.show()

