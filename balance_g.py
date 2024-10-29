import numpy as np 
import DW  

def balance_g(part, N, neighbor):
    """
    计算使用 GSPH 数值近似法的动量和能量平衡的速度、加速度和内能变化。

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

            # i 和 j 粒子之间的空间差
            xij = part['x'][i] - part['x'][k]
            x = abs(xij)  # 空间差的绝对值

            # i 和 j 粒子之间的速度差
            uij = part['u'][i] - part['u'][k]

            # i 和 j 粒子之间的平均密度
            rho = 0.5 * (part['d'][i] + part['d'][k])

            # i 和 j 粒子之间的平均声速
            cij = np.sqrt((part['gamma'] * part['p'][i] / part['d'][i] + part['gamma'] * part['p'][k] / part['d'][k]) / 2)

            # 平滑核梯度（确保使用适合 GSPH 的核函数）
            if x != 0:  # 避免除以零
                dw = DW(2, x, h)
            else:
                dw = 0  # 若 x 为零，设置 dw 为零

            # 动量平衡
            D['u'][i] -= part['m'][k] * (part['p'][i] / part['d'][i] ** 2 + 
                                         part['p'][k] / part['d'][k] ** 2) * dw

            # 能量平衡
            D['e'][i] += 0.5 * part['m'][k] * (part['p'][i] / part['d'][i] ** 2 + 
                                               part['p'][k] / part['d'][k] ** 2) * uij * dw

            # 位置导数
            D['x'][i] = part['u'][i]

    return D
