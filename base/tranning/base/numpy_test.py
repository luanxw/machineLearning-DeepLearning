import numpy as np 
ary = np.array([1,2,3,4,5,6])
print(type(ary), ary , ary.dtype)
b = ary.astype(float)
print(type(b), b , b.dtype)

ary1 = np.array([
    [1,2,3,4],
    [5,6,7,8]
])
#观察维度，size，len的区别
print(ary1.shape, ary1.size, len(ary1))