#coding:utf-8


import tensorflow as tf  
import numpy as np  
import os  
  
#用numpy产生数据  
x_data = np.linspace(-1,1,300)[:, np.newaxis] #转置  
noise = np.random.normal(0,0.05, x_data.shape)  
y_data = np.square(x_data)-0.5+noise  
  
#输入层  
x_ph = tf.placeholder(tf.float32, [None, 1])  
y_ph = tf.placeholder(tf.float32, [None, 1])  
  
#隐藏层  
w1 = tf.Variable(tf.random_normal([1,10]))  
b1 = tf.Variable(tf.zeros([1,10])+0.1)  
wx_plus_b1 = tf.matmul(x_ph, w1) + b1  
hidden = tf.nn.relu(wx_plus_b1)  
  
#输出层  
w2 = tf.Variable(tf.random_normal([10,1]))  
b2 = tf.Variable(tf.zeros([1,1])+0.1)  
wx_plus_b2 = tf.matmul(hidden, w2) + b2  
y = wx_plus_b2  
 
global_step = tf.Variable(0, trainable=False)
#实例化滑动平均类，给衰减率为0.99，当前轮数global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
#ema.apply后的括号里是更新列表，每次运行sess.run（ema_op）时，对更新列表中的元素求滑动平均值。
#在实际应用中会使用tf.trainable_variables()自动将所有待训练的参数汇总为列表
#ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())

 
#损失  
loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_ph-y),reduction_indices=[1]))  
#train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss,global_step=global_step)  
  
#保存模型对象saver  
saver = tf.train.Saver()  
MODEL_SAVE_PATH = 'tmp/'   
#MODEL_NAME = "mymodel.ckpt"
MODEL_NAME = "mymodel"


#判断模型保存路径是否存在，不存在就创建  
if not os.path.exists('tmp/'):  
    os.mkdir(MODEL_SAVE_PATH)  
 
#初始化  
with tf.Session() as sess:  
    #判断模型是否存在  
    if os.path.exists(MODEL_SAVE_PATH +'/checkpoint'):         
         #存在就从模型中恢复变量  
        saver.restore(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME) )   
        print(sess.run(w1)) 
        print(sess.run([w1, ema.average(w1)]))
    #不存在就初始化变量   
    else:  
        init = tf.global_variables_initializer() 
        sess.run(init)  

    for i in range(1000):  
        _,loss_value = sess.run([train_op,loss], feed_dict={x_ph:x_data, y_ph:y_data})  
        if(i%50==0): 
            #保存
            save_path = saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME) )  
            print("迭代次数：%d , 训练损失：%s"%(i, loss_value))  
