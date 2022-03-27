import numpy as np
from tensorflow import keras
import tensorflow as tf
import sklearn.datasets as datasets
import matplotlib.pyplot as mp

print('tnsorflow版本:',tf.__version__)
BATCH_SIZE = 36

X,Y = datasets.make_moons(200,noise=0.10)
print(X.shape, Y.shape)
# print('y: ',y)

Y = np.array(np.column_stack((Y, ~Y+2)),dtype='f4')
# print('y: ',y)
# 定义神经网络的输入、参数、输出、定义前向传播过程 
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32,shape= (None,2),name= 'x')
y = tf.compat.v1.placeholder(tf.float32,shape= (None,1),name= 'y')


w1 =  tf.Variable(tf.random.normal([2,3],stddev=1,seed=1))
b1 =  tf.Variable(tf.random.normal([3,],stddev=1,seed=1))
w2 =  tf.Variable(tf.random.normal([3,2],stddev=1,seed=1)) 
b2 =  tf.Variable(tf.random.normal([2,],stddev=1,seed=1)) 

#定义正向传播
l1 = tf.nn.sigmoid(tf.add(tf.matmul(x,w1),b1))
predict_y =  tf.add(tf.matmul(l1,w2),b2)

'''
softmax_cross_entropy_with_logits
    1.将logits转换成概率
    2.计算交叉熵损失

tf.losses.softmax_cross_entropy()： 用tf.nn.softmax_cross_entropy_with_logits()实现。
tf.losses.sparse_softmax_cross_entropy()： 用tf.nn.sparse_softmax_cross_entropy_with_logits()实现。
tf.losses.sparse_softmax_cross_entropy() 等价于tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits)，输入labels是非one-hot编码格式。

'''
#定义损失函数、以及反向传播方法

# loss = tf.reduce_sum(tf.pow(predict_y - Y, 2))/(2 * X.shape)
loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=predict_y))
# loss = tf.losses.sparse_softmax_cross_entropy(labels=Y,logits=predict_y)

# 定义优化器
# train_step = tf.train().GradientDesentOptimier(0.01).minimize(loss)
# train_step = tf.train().AdamOptimier(0.01).minimize(loss)
train_step = tf.keras.optimizers.Adam(lr = 0.01).minimize(loss, var_list=[w1,b1,w2,b2])


with tf.Session() as sess:
    init_global = tf.global_variables_initializer()
    sess.run(init_global)

    #训练模型
    STEPS = 3000
    for step in STEPS:
        start = (step * BATCH_SIZE)%32
        end = start + BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end]})
        if i % 300 == 0 :
            total_loss = sess.run(loss, feed_dict={x:X, y:Y})
            print('第 %d 轮训练之后, 损失函数值: %f',step, total_loss)
predict_y =  sess.run(predict_y,feed_dict={x:X})
predict_y = np.piecewise(predict_y, [predict_y < 0 ,predict_y > 0],[0,1])


l , r = X[:,0].min()-1,  X[:,0].max()+1
b ,t  = X[:,1].min()-1,  X[:,1].max()+1
n = 500
grid_x , grid_y = np.meshgrid(np.linspace(l,r,n), np.linspace(b,t,n))
sample = np.column_stack(grid_x.ravel() , grid_y.ravel())
grid_z = sess.run(predict_y,feed_dict={x:sample})
grid_z =  grid_z.reshape(-1,2)[:,0]
grid_z =  np.piecewise(grid_z, [grid_z < 0, grid_z >0],[0,1])
grid_z =  grid_z.reshape(grid_x.shape)
mp.figure('logistic Classifiction', facecolor='light')
mp.title('logistic Classifiction', fontsize=20)
mp.xlabel('x', fontsize=20)
mp.ylabel('y', fontsize=20)
mp.tick_params(labelsize= 10)
mp.pcolormesh(grid_x,grid_y,grid_z,cmap='gray')
mp.scatter(X[:,0],X[:,1],c=Y[:,0], cmap='brg')
mp.show()
