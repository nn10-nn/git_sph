import numpy as np 
import matplotlib.pyplot as plt
from scipy.io import loadmat
import time
import pandas as pd
from startSodTube import preProcess  
import neighborhood
import timestep
import density
import balance_g   
import integration 
import plotSodTubeResult
import DW
from Wf import W

# 初始数据设定
n = 80                             # 最小粒子数
N = 9 * n                     # 总粒子数
dx1 = 0.5 / (8 * n)             # 左侧空间步长  这里是8,一共有9n个粒子，所以密度比0.1和0.125是8比1
dx2 = 0.5 / n                 # 右侧空间步长
gamma = 1.4                        # 气体常数
part = preProcess(N, dx1, dx2, gamma)    # 初始化粒子
T = 0.24                           # 终止时间
t = 0.0                             # 初始时间
k = 0.0                            # 初始交互计数

# 从csv文件加载分析解
data1= pd.read_csv('output.csv')

# 初始化变量
D = {'u': np.ones(N)}
# 主循环
while t < T:
    
    # 第一步 
    neighbor = neighborhood.neighborhood(part, N)  # 计算邻域
    rho = density.density(part, N, neighbor) # 计算密度
    part['d'] = rho                         # 更新临时粒子的密度
    
    # 更新压力和声速的计算
    part['p'] = (gamma - 1) * part['d'] * part['e']  # 使用临时粒子的能量
    part['c'] = np.sqrt(gamma * part['p'] / part['d'])  # 基于最新压力和密度计算声速
    part,dt =balance_g.balance_g(part, N, neighbor)



    # 边界条件
    q = 170
    m = 30
    part['d'][0:q] = 1                       # 更新左侧边界的密度
    part['p'][0:q] = 1                       # 更新左侧边界的压力
    part['u'][0:q] = 0                       # 更新左侧边界的速度
    part['e'][0:q] = gamma                    # 更新左侧边界的能量
    part['d'][N-m-1:N] = 0.125                  # 更新右侧边界的密度
    part['p'][N-m-1:N] = 0.1               # 更新右侧边界的压力
    part['u'][N-m-1:N] = 0                      # 更新右侧边界的速度
    part['e'][N-m-1:N] = gamma * 0.1 / 0.125                 # 更新右侧边界的能量


    # 错误检测
    if np.isnan(part['d']).any() or np.isnan(part['p']).any() or np.isnan(part['u']).any() or np.isnan(part['e']).any() or np.isnan(part['c']).any():
        print(part['e'])
        raise ValueError(f'Error! NaN value detected (Iteration {k}, time: {t:.3f})')
    else:
        t += dt 
        k += 1
    
    # 在主循环的最后添加打印语句
    #print(f"Time: {t:.3f}, Average Pressure: {np.mean(part['p'])}, Average Density: {np.mean(part['d'])}, Average Energy: {np.mean(part['e'])}, Average Velocity: {np.mean(part['u'])}")

# 绘图
plotSodTubeResult.plotSodTubeResults(part, data1, 0)
