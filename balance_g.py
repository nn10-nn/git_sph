import numpy as np
import DW  

gamma = 1.4  # 气体常数

def roe_flux(uL, uR, pL, pR, dL, dR, gamma):
    """
    使用 Roe 格式的黎曼求解器计算左右粒子之间的速度和压力.

    参数:
    uL, uR (float): 左、右粒子的速度
    pL, pR (float): 左、右粒子的压力
    dL, dR (float): 左、右粒子的密度
    gamma (float): 气体常数

    返回:
    tuple: 返回中间状态的速度 u_star 和压力 p_star
    """
    # 左右声速
    cL = np.sqrt(gamma * pL / dL)
    cR = np.sqrt(gamma * pR / dR)
    # 声速平均
    cLR = 0.5 * (cL + cR)

    #print(f"cL = {cL}, cR = {cR}, cLR = {cLR}")

    # 使用 Roe 格式公式计算中间状态的速度和压力
    u_star = 0.5 * (uL + uR - (pR - pL) / cLR)
    p_star = 0.5 * (pL + pR - cLR * (uR - uL))

    #print(f"u_star = {u_star}, p_star = {p_star}")

    return u_star, p_star

def balance_g(part, N, neighbor):
    """
    使用 GSPH 方法和 Roe 格式黎曼求解器计算动量和能量平衡。

    参数:
    part (dict): 包含粒子属性的字典。
    N (int): 粒子总数。
    neighbor (list): 每个粒子的邻居信息列表。

    返回:
    dict: 包含粒子属性导数的字典。
    """
    D = {'u': np.zeros(N), 'e': np.zeros(N), 'x': np.zeros(N)}

    for i in range(N):
        for j in range(neighbor[i][0]):
            k = neighbor[i][j + 1]  # 邻居索引

            # 计算 i 和 j 粒子之间的平均平滑长度
            h = 0.5 * (part['h'][i] + part['h'][k])

            # 计算 i 和 j 粒子之间的空间差
            xij = part['x'][i] - part['x'][k]
            x = abs(xij)  # 空间差的绝对值

            # 获取 i 和 j 粒子的物理量
            uL, uR = part['u'][i], part['u'][k]
            pL, pR = part['p'][i], part['p'][k]
            dL, dR = part['d'][i], part['d'][k]

            # 使用 Roe 黎曼求解器计算中间状态的速度和压力
            u_star, p_star = roe_flux(uL, uR, pL, pR, dL, dR, gamma)

           #print(f"Particle {i}, Neighbor {k}: u_star = {u_star}, p_star = {p_star}")

            # 计算平滑核梯度
            if x != 0:  # 避免除以零
                dw = DW.DW(2, x, h)  # 使用二次核
            else:
                dw = 0  # 若 x 为零，设置 dw 为零
            
            #print(f"Particle {i}, Neighbor {k}: x = {x}, h = {h}, dw = {dw}")


            # 动量平衡：使用中间状态的压力
            D['u'][i] -= part['m'][k] * (p_star / dL**2 + p_star / dR**2) * dw

            #print(f"Particle {i}: D['u'][i] = {D['u'][i]}")  # 调试打印动量导数

            # 改进后的能量平衡计算
            D['e'][i] += 0.5 * part['m'][k] * (u_star ** 2 + p_star / dL - (uL ** 2 + pL / dL)) * dw
 
            #print(f"Particle {i}: D['e'][i] = {D['e'][i]}")  # 调试打印能量导数

            # 位置导数
            D['x'][i] = part['u'][i]

    return D
