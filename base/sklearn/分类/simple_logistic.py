from cProfile import label
from turtle import color
import numpy as np
import matplotlib.pylab as mp

x = np.array([
    [3, 1],
    [2, 5],
    [1, 8],
    [6, 4],
    [5, 2],
    [3, 5],
    [4, 7],
    [4, -1]])
y = np.array([0, 1, 1, 0, 0, 1, 1, 0])
# 绘制分类边界线
l,r = x[:,0].min()-1, x[:,0].max()+1
b,t = x[:,1].min()-1, x[:,1].max()+1
# 把可视空间划分为 500 * 500
n=500 
grid_x,grid_y = np.meshgrid(np.linspace(l,r,n), np.linspace(b,t,n))
grid_z = np.piecewise(grid_x,[grid_x>grid_y,grid_x<grid_y],[1,0])

mp.figure('Simple Classifiation', facecolor='lightgray')
mp.title('Simple Classifiation', fontsize=20)
mp.scatter(x[:,0],x[:,1],c=y,cmap='jet',label='simple points',zorder=3)
mp.pcolormesh(grid_x,grid_y,grid_z,cmap='gray')
mp.legend()
mp.show()