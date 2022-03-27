'''
独热编码(（One-Hot Encoding）)
为样本特征的每个值建立一个由一个1和若干个0组成的序列，用该序列对所有的特征值进行编码。
即: 

特征并不总是连续值，而有可能是分类值。
离散特征的编码分为两种情况：
　　1、离散特征的取值之间没有大小的意义，比如color：[red,blue],那么就使用one-hot编码
　　2、离散特征的取值有大小的意义，比如size:[X,XL,XXL],那么就使用数值的映射{X:1,XL:2,XXL:3}
pandas 和 sklearn都可以实现该功能
参考地址: https://www.cnblogs.com/zongfa/p/9305657.html

独热编码相关API：

# 创建一个独热编码器
# sparse： 是否使用紧缩格式（稀疏矩阵）
# dtyle：  数据类型
ohe = sp.OneHotEncoder(sparse=是否采用紧缩格式, dtype=数据类型)
# 对原始样本矩阵进行处理，返回独热编码后的样本矩阵。
result = ohe.fit_transform(原始样本矩阵)
或:
ohe = sp.OneHotEncoder(sparse=是否采用紧缩格式, dtype=数据类型)
# 对原始样本矩阵进行训练，得到编码字典
encode_dict = ohe.fit(原始样本矩阵)
# 调用encode_dict字典的transform方法 对数据样本矩阵进行独热编码
result = encode_dict.transform(原始样本矩阵)

'''
import numpy as np
import sklearn.preprocessing as sp

raws_samples = np.array([
    [13,45,7567],
    [23,67,2452],
    [13,33,9732]
])
print(raws_samples)
# 创建一个独热编码器
# sparse： 是否使用紧缩格式（稀疏矩阵）
# dtyle：  数据类型
ohe = sp.OneHotEncoder(sparse=False,dtype=int)
ohe_dict = ohe.fit(raws_samples)
ohe_dict_transform = ohe_dict.transform(raws_samples)
print('----------ohe_dict_transform--------------')
print(ohe_dict_transform)
# 直接更新编码器
ohe_transform =  ohe.fit_transform(raws_samples)
print('----------ohe_transform--------------')
print(ohe_transform)
