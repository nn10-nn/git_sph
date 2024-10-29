import numpy as np
import math

  
def neighborhood(part, N):
    """
    查找每个粒子最近的邻居，属于以该粒子为中心的局部或紧凑域。
    
    参数:
    part (dict): 包含粒子属性的字典，包含以下键:
                 - x: 粒子位置数组
                 - h: 粒子平滑长度数组
    N (int): 粒子总数
    alpha (array): SPH参数
    
    返回:
    numpy.ndarray: 包含邻居数量和标识的数组, 第一列为邻居数量, 其余列为邻居ID
    """
# 初始化邻居数组，第一列存储邻居数量，其他列存储邻居ID
    neighbor = np.zeros((N, N), dtype=int)  # 最大邻居数设置为 N
    
    for i in range(N):
        neighbor_count = 0  # 记录当前粒子的邻居数量
        for j in range(N):
            if i != j:
                h = (part['h'][i] + part['h'][j]) / 2
                distance = abs(part['x'][i] - part['x'][j])
                # 如果距离在支持域内，则认为是邻居
                if distance <= h:  
                    neighbor[i, neighbor_count + 1] = j  # 存储邻居ID
                    neighbor_count += 1  # 增加邻居计数
        neighbor[i, 0] = neighbor_count  # 将邻居数量存储在第一列

    return neighbor

