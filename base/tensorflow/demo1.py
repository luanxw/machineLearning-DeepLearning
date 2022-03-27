from statistics import stdev
import numpy as np
import tensorflow as tf
import sklearn.datasets as sd
import matplotlib.pyplot as mp

BATCH_SIZE = 8
seed = 2353

#基于seed产生随机数
rng = np.random.seed(0)
#随机数返回32 X 2的矩阵、表述32行宽度和长度作为输入数据集
x = rng.rand(32,2)
# 从32行2列的数据中、去处一行、如果和大于1,则给y赋值为0,和小于1,则赋值为1作为输入数据集的标签
y = [[int (x0 +x1 < 1)] for (x0,x1) in x]
print('x: ',x)
print('y: ',y)
# 定义神经网络的输入、参数、输出、定义前向传播过程 
x = tf.placeholder(tf.float32,shape= (None,2))
y_ = tf.placeholder(tf.float32,shape= (None,1))

w1 =  tf.variable(tf.random_normal([2,3],stddev=1,seed=1))
w2 =  tf.variable(tf.random_normal([3,1],stddev=1,seed=1))