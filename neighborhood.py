import numpy as np
import math

  
def neighborhood(part, N, alpha):
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
    R = N / alpha[3]
    # 初始化邻居数组，第一列存储邻居数量，其他列存储邻居ID
    neighbor = np.zeros((N, min(N, N + math.ceil(R) )), dtype=int)
    
    # 定义搜索半径
    
    for i in range(N):
        for j in range(max(0, i - math.ceil(R) - 1), min(N - 1, i + math.ceil(R) - 1)):  # 半径位移
            if i != j:
                h = (part['h'][i] + part['h'][j]) / 2
                if abs(part['x'][i] - part['x'][j]) <= alpha[4] * h:  # 支持域
                    neighbor[i, 0] += 1  # 邻居数量
                    neighbor[i, neighbor[i, 0]] = j  # 确保 j 已经被定义或正确计算  
    
    return neighbor

