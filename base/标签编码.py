# -*- coding: utf-8 -*-

'''
根据字符串形式的特征值在特征序列中的位置，为其指定一个数字标签，用于提供给基于数值算法的学习模型。

标签编码相关API：

# 获取标签编码器
lbe = sp.LabelEncoder()
# 调用标签编码器的fit_transform方法训练并且为原始样本矩阵进行标签编码
result = lbe.fit_transform(原始样本矩阵)
# 根据标签编码的结果矩阵反查字典 得到原始数据矩阵
samples = lbe.inverse_transform(result)


'''
import numpy as np
import sklearn.preprocessing as sp 

raws_sample = np.array(["audi", "ford", "audi", "toyota",
                        "ford", "bmw", "ford", "redflag", "audi"])
print(raws_sample)
lable = sp.LabelEncoder()
lable_sample = lable.fit_transform(raws_sample)
print(lable_sample)
inverse_sample = lable.inverse_transform(lable_sample)
print(inverse_sample)