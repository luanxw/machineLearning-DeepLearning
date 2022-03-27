# -*-coding:utf-8-*-

import numpy as np
import sklearn.preprocessing as sp

'''
numpy:
    NumPy 是 Python 中科学计算的基础包。它是一个 Python 库，提供多维数组对象、各种派生对象（例如掩码数组和矩阵）以及用于对数组进行快速操作的各种例程，
包括数学、逻辑、形状操作、排序、选择、I/O 、离散傅里叶变换、基本线性代数、基本统计运算、随机模拟等等。

sklearn:
    scikit-learn是基于Python语言的机器学习库，具有：
        简单高效的数据分析工具
        可在多种环境中重复使用
        建立在Numpy，Scipy以及matplotlib等数据科学库之上
        开源且可商用的-基于BSD许可

其中:sklearn.preprocessing函数预处理包,还可以进行监督学习、无监督学习、模型评估检验、数据集转换(特征处理、预处理数据、标签转换等)
'''


'''
本代码块学习目的: 

    由于一个样本的不同特征值差异较大，不利于使用现有机器学习算法进行样本处理。**均值移除**可以让样本矩阵中的每一列的平均值为0，标准差为1。
sklearn.preprocessing函数方法:
    scale() 每列的数据均值变成0，标准差变为1,具体可通过axis属性配置
    mean() 求均值, ,axis=0计算每一列均值; axis=1 计算每一行的均值 ;不填 计算全局均值
    std() 计算标准差,axis=0计算每一列的标准差; axis=1 计算每一行的标准差 ;不填 计算全局标准差 
numpy 函数方法:
    np.average()   求平均值
'''
data_example = np.array([
    [17,28,8909],
    [34,45,7686],
    [24,34,6544]])
# print(data_example.std(axis=1))# axis=0计算每一列的标准差; axis=1 计算每一行的标准差 ;不填 计算全局标准差 
# print(data_example.std())  # 标准偏差是与平均值的平方偏差的平均值的平方根，即，其中 。std = sqrt(mean(x))x = abs(a - a.mean())**2
# print(data_example.mean(axis=0))
# print(np.average(data_example))
# print('----------------------------------------')
# std_example = sp.scale(data_example)
# print(std_example)
# print(std_example.mean())
# print(std_example.mean(axis=0))
# print(std_example.std(axis=0)) 
# print(std_example.std())
print(data_example[1:])