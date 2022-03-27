'''
有些情况每个样本的每个特征值具体的值并不重要，但是每个样本特征值的占比更加重要。
所以归一化即:用每个样本的每个特征值除以该样本各个特征值绝对值的总和。变换后的样本矩阵，每个样本的特征值绝对值之和为1。


归一化相关API
    # array 原始样本矩阵
    # norm  范数
    #    l1 - l1范数，向量中个元素绝对值之和
    #    l2 - l2范数，向量中个元素平方之和
    # 返回归一化预处理后的样本矩阵
    sp.normalize(array, norm='l1')
'''
import numpy as np
import sklearn.preprocessing as sp

raw_data = np.array([
    [17.,45.,5456.],
    [20.,43.,9834.],
    [56.,34.,3683.]])
print(raw_data)
raw_example = raw_data.copy()
# for row in raw_example:
#     row /= abs(row).sum()

# for row in raw_example:
#     print(abs(row))
#     row /= abs(row).sum()
# print(raw_example)
# print(abs(raw_example).sum(axis=1))
#归一化处理

raw_normal = sp.normalize(raw_example,norm='l1')
print(raw_normal)

raw_normal2 = sp.normalize(raw_example,norm='l2')
print(raw_normal2)