# -*- coding: utf-8 -*-

'''
所谓模型训练，就是根据已知的x和y，找到最佳的模型参数w0 和 w1，尽可能精确地描述出输入和输出的关系。
'''

from matplotlib import projections
import numpy as np
import matplotlib.pyplot as mp

tran_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
tran_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])
test_x = np.array([0.45, 0.55, 1.0, 1.3, 1.5])
test_y = np.array([4.8, 5.3, 6.4, 6.9, 7.3])

times =  1000 # 定义梯度下降次数
lrate = 0.01 # 记录每次梯度下降参数变化率
epoches = []
w0, w1, losser = [1],[1],[]
for i in range(1, times + 1):
    epoches.append(i)
    loss = (((w0[-1]+w1[-1]*tran_x) - tran_y) ** 2).sum() / 2
    losser.append(loss)
    d0 = ((w0[-1] + w1[-1] * tran_x) - tran_y).sum()
    d1 = (((w0[-1] + w1[-1] * tran_x) - tran_y) * tran_x).sum()
    print('{:4}> w0={:.8f}, w1={:.8f}, loss={:.8f}'.format(epoches[-1], w0[-1], w1[-1], losser[-1]))
    w0.append(w0[-1] - d0 * lrate)
    w1.append(w1[-1] - d1 * lrate)

pred_test_y = w0[-1] + w1[-1]*test_x
print(pred_test_y)

mp.figure('Training Progress', facecolor='lightgray')
mp.title('Training Progress', fontsize='18')
mp.subplot(311)
mp.ylabel('w0')
mp.grid()
mp.plot(epoches,w0[:-1],label='w0') 
mp.legend()

mp.subplot(312)
mp.ylabel('w1')
mp.grid()
mp.plot(epoches,w1[:-1],label='w1') 
mp.legend()

mp.subplot(313)
mp.ylabel('losser')
mp.grid()
mp.plot(epoches,losser,label=r'$losser$') 
mp.legend()
mp.tight_layout()

mp.show()

#绘制三维图、并显示梯度变化
n =500
w0_grid, w1_grid = np.meshgrid(
    np.linspace(-3,10,n),
    np.linspace(-3,10,n)
    )
loss_grid = 0
for x,y in zip(tran_x,tran_y):
    loss_grid += (w0_grid + w1_grid * x - y) ** 2 / 2
mp.figure('loss function')
ax3d = mp.gca()
ax3d.set_xlabel('w0')
ax3d.set_ylabel('w1')
ax3d.set_zorder('loss')
ax3d.plo