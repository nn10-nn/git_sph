import numpy as np
from Wf import W

def density(part, N, neighbor):
    """
    计算基于局部粒子数量的粒子密度。

    参数:
    part (dict): 包含粒子属性的字典，如位置 'x', 平滑长度 'h', 质量 'm'
    N (int): 粒子总数
    neighbor (list of lists): 每个粒子的邻居数量和标识
    alpha (list): 纠正参数，alpha[2]用于选择平滑核

    返回:
    rho (ndarray): 平滑后的密度数组
    """
    
    rho = np.zeros(N)  # 密度向量预分配

    for i in range(N):
        for j in range(neighbor[i][0]):
            k = neighbor[i][j+1]  # 邻居项
            
            # i 和 j 粒子之间的平均平滑长度
            h = 0.5 * (part['h'][i] + part['h'][k])
            
            # i 和 j 粒子之间的空间差的绝对值
            x = abs(part['x'][i] - part['x'][k])
            
            # 平滑核
            w = W(2, x, h)
            
            # 以SPH术语对密度进行累加
            rho[i] += part['m'][k] * w

    # 密度校正
    rho += 0.0855
    
    return rho
