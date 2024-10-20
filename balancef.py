import numpy as np
import DW

def balancef(part, N, neighbor, alpha):
    """
    计算使用 SPH 数值近似法的动量和能量平衡的速度、加速度和内能变化。

    参数:
    part (dict): 包含粒子属性的字典。
    N (int): 粒子总数。
    neighbor (list): 每个粒子的邻居信息列表。
    alpha (list): 校正参数列表。

    返回:
    dict: 包含粒子属性导数的字典。
    """
    D = {'u': np.zeros(N), 'e': np.zeros(N), 'x': np.zeros(N)}

    for i in range(N):
        for j in range(neighbor[i][0]):
            k = neighbor[i][j + 1]  # 邻居索引

            # i 和 j 粒子之间的平均平滑长度
            h = 0.5 * (part['h'][i] + part['h'][k])

            # i 和 j 粒子之间的空间差的绝对值
            x = abs(part['x'][i] - part['x'][k])

            # i 和 j 粒子之间的空间差
            xij = part['x'][i] - part['x'][k]

            # i 和 j 粒子之间的速度差
            uij = part['u'][i] - part['u'][k]

            # i 和 j 粒子之间的平均密度
            rho = 0.5 * (part['d'][i] + part['d'][k])

            # i 和 j 粒子之间的平均声速
            cij = 0.5 * (part['c'][i] + part['c'][k])

            # 平滑核梯度
            dw = xij * DW.DW(alpha[2], x, h) / (x * h)

            # 人工粘性
            if uij * xij < 0:
                muij = -(uij * xij / h) / ((x / h) ** 2 + 0.01)
                IIij = (alpha[0] * cij * muij + alpha[1] * muij ** 2) / rho
            else:
                IIij = 0

            # 动量平衡
            D['u'][i] -= part['m'][k] * (part['p'][i] / part['d'][i] ** 2 + 
                                         part['p'][k] / part['d'][k] ** 2 + IIij) * dw

            # 能量平衡
            D['e'][i] += 0.5 * part['m'][k] * (part['p'][i] / part['d'][i] ** 2 + 
                                               part['p'][k] / part['d'][k] ** 2 + IIij) * uij * dw

            # 位置导数
            D['x'][i] = part['u'][i]

    return D
