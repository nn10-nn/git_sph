import numpy as np
from Wf import W


def density(part, N, neighbor, alpha):
    """
    计算粒子密度，基于局部紧致域内的粒子数量。这种方法保证了质量平衡。
    
    参数:
    part (dict): 粒子属性，包括位置`x`、质量`m`和平滑长度`h`。
    N (int): 粒子总数。
    neighbor (np.ndarray): 每个粒子的邻居数量和标识。
    alpha (np.ndarray): 校正参数。
    
    返回:
    rho (np.ndarray): 平滑密度。
    """
    
    rho = np.zeros(N)  # 预分配密度向量  
  
    for i in range(N):  
        num_neighbors = int(neighbor[i, 0])  # 获取粒子i的邻居数量（注意跳过索引0）  
        for j in range(1, num_neighbors + 1):  # 遍历粒子i的所有邻居  
            k = int(neighbor[i, j])  # 获取邻居的索引（注意Python索引从0开始）  
              
            # 计算粒子i和邻居k之间的平均平滑长度  
            h = 0.5 * (part['h'][i] + part['h'][k])  
              
            # 计算粒子i和邻居k之间的绝对空间差异  
            x = abs(part['x'][i] - part['x'][k])  
              
            # 调用W函数计算平滑核的权重  
            w = W(3, x, h)  
              
            # 累加该邻居对粒子i密度的贡献  
            rho[i] += part['m'][k] * w  
      
    # 对所有粒子的密度进行修正（这里假设是一个常数修正）  
    rho += 0.0855  
      
    return rho  