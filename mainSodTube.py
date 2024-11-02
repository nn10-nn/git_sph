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
N = 5 * n                          # 总粒子数
dx1 = 0.6 / (4 * n)                # 左侧空间步长
dx2 = 0.6 / n                      # 右侧空间步长
gamma = 1.4                        # 气体常数
part = preProcess(N, dx1, dx2, gamma)    # 初始化粒子
T = 0.24                           # 终止时间
t = 0.0                             # 初始时间
k = 0.0                            # 初始交互计数

# 从csv文件加载分析解
data1= pd.read_csv('data1.csv')

# 初始化变量
D = {'u': np.ones(N)}

# 主循环
while t < T:
    
    # 时间积分的第一步 (RK2)
    neighbor = neighborhood.neighborhood(part, N)  # 计算邻域
    dt = timestep.timestep(part, N)         # 计算时间步长
    rho = density.density(part, N, neighbor)  # 计算密度
    ipart = part.copy()                      # 复制当前粒子状态
    ipart['d'] = rho                         # 更新临时粒子的密度
    
    # 修改：更新压力和声速的计算
    ipart['p'] = (gamma - 1) * ipart['d'] * ipart['e']  # 使用临时粒子的能量
    ipart['c'] = np.sqrt(gamma * ipart['p'] / ipart['d'])  # 基于最新压力和密度计算声速

    D = balance_g.balance_g(part, N, neighbor)   # 计算平衡力
    v, e, x = integration.integration(1, part, ipart, N, D, dt)    # 执行积分步骤1
    ipart['x'] = x                           # 更新临时粒子的位置
    ipart['u'] = v                           # 更新临时粒子的速度
    ipart['e'] = e                           # 更新临时粒子的能量

    # 时间积分的第二步 (RK2)
    t += dt / 2
    neighbor = neighborhood.neighborhood(ipart, N) # 计算邻域
    rho = density.density(ipart, N, neighbor) # 计算密度
    part['d'] = rho                          # 更新粒子的密度
    
    # 修改：更新压力和声速的计算
    part['p'] = (gamma - 1) * part['d'] * ipart['e']  # 计算压力
    part['c'] = np.sqrt(gamma * part['p'] / part['d'])  # 使用最新压力和密度计算声速

    D = balance_g.balance_g(ipart, N, neighbor)  # 计算平衡力
    v, e, x = integration.integration(2, part, ipart, N, D, dt)   # 执行积分步骤2
    part['x'] = x                           # 更新粒子的位置
    part['u'] = v                           # 更新粒子的速度
    part['e'] = e                           # 更新粒子的能量

    # 边界条件
    q = 170
    m = 30
    part['d'][:q] = 1                       # 更新左侧边界的密度
    part['p'][:q] = 1                       # 更新左侧边界的压力
    part['u'][:q] = 0                       # 更新左侧边界的速度
    part['e'][:q] = 2.5                     # 更新左侧边界的能量
    part['d'][-m:] = 0.25                   # 更新右侧边界的密度
    part['p'][-m:] = 0.1795                 # 更新右侧边界的压力
    part['u'][-m:] = 0                      # 更新右侧边界的速度
    part['e'][-m:] = 1.795                  # 更新右侧边界的能量


    # 错误检测
    if np.isnan(part['d']).any() or np.isnan(part['p']).any() or np.isnan(part['u']).any() or np.isnan(part['e']).any() or np.isnan(part['c']).any():
        print(e)
        raise ValueError(f'Error! NaN value detected (Iteration {k}, time: {t:.3f})')
    else:
        t += dt / 2
        k += 1
    
    # 在主循环的最后添加打印语句
    print(f"Time: {t:.3f}, Average Pressure: {np.mean(part['p'])}, Average Density: {np.mean(part['d'])}, Average Energy: {np.mean(part['e'])}, Average Velocity: {np.mean(part['u'])}")

    # 绘图
    plotSodTubeResult.plotSodTubeResults(part, data1, 0)
