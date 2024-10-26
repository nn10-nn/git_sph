import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import time
from startSodTube import preProcess
import neighborhood
import timestep
import density
import balancef
import integration
import plotSodTubeResult
import DW
from Wf import W

def roeRiemannSolver(pl, pr, ul, ur, rhol, rhor, gamma):
    # 根据提供的公式计算加权声速
    clr = np.sqrt((gamma * pl + gamma * pr) / (rhol + rhor))
    # 计算中心压力和中心速度
    p_star = 0.5 * (pl + pr - clr* (ur - ul) )
    u_star = 0.5 * ( (ul + ur) - (pr - pl) / clr )
    return p_star, u_star

# 初始数据设定
n = 80
N = 5 * n
dx1 = 0.6 / (4 * n)
dx2 = 0.6 / n
gamma = 1.4
part = preProcess(N, dx1, dx2, gamma)  
T = 0.24
t = 0.0
k = 0.0

# 从csv文件加载分析解
data1 = pd.read_csv('data1.csv')

# ALPHA参数向量
alpha = [1, 2, 4, 20, 2]

# 初始化变量
D = {'u': np.ones(N)}

# 主循环
while t < T:
    # 时间积分的第一步 (RK2)
    neighbor = neighborhood.neighborhood(part, N, alpha)
    dt = timestep.timestep(part, N, alpha, D)
    rho = density.density(part, N, neighbor, alpha)
    ipart = part.copy()
    ipart['d'] = rho
    ipart['p'] = (gamma - 1) * ipart['d'] * part['e']
    ipart['c'] = np.sqrt((gamma - 1) * part['e'])
    D = balancef.balancef(part, N, neighbor, alpha)
    
    # 更新粒子状态，使用 Roe 黎曼求解器
    for i in range(N):
     num_neighbors = neighbor[i, 0]
    for n in range(1, num_neighbors + 1):
        j = neighbor[i, n]
        # 使用粒子 i 和其邻居 j 的数据
        pl = ipart['p'][i]
        pr = ipart['p'][j]
        ul = ipart['u'][i]
        ur = ipart['u'][j]
        rhol = ipart['d'][i]
        rhor = ipart['d'][j]
        p_star, u_star = roeRiemannSolver(pl, pr, ul, ur, rhol, rhor, gamma)
        ipart['p'][i] = p_star
        ipart['u'][i] = u_star

    v, e, x = integration.integration(1, part, ipart, N, D, dt)
    ipart['x'] = x
    ipart['u'] = v
    ipart['e'] = e   
    print (t)
    # 时间积分的第二步 (RK2)
    t += dt / 2
    neighbor = neighborhood.neighborhood(ipart, N, alpha)
    rho = density.density(ipart, N, neighbor, alpha)
    part['d'] = rho
    part['p'] = (gamma - 1) * part['d'] * ipart['e']
    part['c'] = np.sqrt((gamma - 1) * ipart['e'])
    D = balancef.balancef(ipart, N, neighbor, alpha)
    v, e, x = integration.integration(2, part, ipart, N, D, dt)
    part['x'] = x
    part['u'] = v
    part['e'] = e

    # 边界条件
    q = 170
    m = 30
    part['d'][:q] = 1
    part['p'][:q] = 1
    part['u'][:q] = 0
    part['e'][:q] = 2.5
    part['d'][-m:] = 0.25
    part['p'][-m:] = 0.1795
    part['u'][-m:] = 0
    part['e'][-m:] = 1.795

    # 错误检测
    if np.isnan(part['d']).any() or np.isnan(part['p']).any() or np.isnan(part['u']).any() or np.isnan(part['e']).any() or np.isnan(part['c']).any():
        print(e)
        raise ValueError(f'Error! NaN value detected (Iteration {k}, time: {t:.3f})')
    else:
        t += dt / 2
        k += 1
    if t >= T:
        break
# 绘图
plotSodTubeResult.plotSodTubeResults(part, data1, 0)
