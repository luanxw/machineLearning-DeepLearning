# 将样本矩阵中的每一列的最小值和最大值设定为相同的区间，统一各列特征值的范围。一般情况下会把特征值缩放至[0, 1]区间。
# 降低数据在机器学习中的影响对于求相似度之类的好用

import numpy as np
## 解决机器学习中的科学计算问题的包
import sklearn.preprocessing as sp

# NumPy 是 Python 中科学计算的基础包。它是一个 Python 库，提供多维数组对象、各种派生对象（例如掩码数组和矩阵）以及用于对数组进行快速操作的各种例程，
# 包括数学、逻辑、形状操作、排序、选择、I/O 、离散傅里叶变换、基本线性代数、基本统计运算、随机模拟等等。
# 

data_example = np.array([
    [23.,234.,2343.],
    [26.,345.,4565.],
    [32.,596.,5675.]
])
print(data_example)
mms_samples = data_example.copy()
for col in mms_samples.T:
    col_min = col.min()
    col_max = col.max()
    a = np.array([
        [col_min, 1],
        [col_max, 1]])
    b = np.array([0, 1])
    x = np.linalg.solve(a, b) #方程组ax=b的解
    col *= x[0]
    col += x[1]
print(mms_samples)
nms = sp.MinMaxScaler(feature_range=(0,1))
nms_result = nms.fit_transform(data_example)
print(nms_result)
