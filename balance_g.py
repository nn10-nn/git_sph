import numpy as np
import timestep
import Wf
import DW  
import density


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
    dt = timestep.timestep(part, N)
    rho = density.density(part, N, neighbor)  # 计算密度
    ipart = part.copy()     #更新ipart
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

            # 计算平滑核梯度
            if x != 0:  # 避免除以零
                dw = DW.DW(2, x, h)  # 使用二次核
            else:
                dw = 0  # 若 x 为零，设置 dw 为零


             # 计算高斯核函数值
            W_ab = Wf.W(2,x, h)  # 使用缩放后的核函数
            Vab_squared = (1 / dL / dL ) * W_ab + (1 / dR / dR) * W_ab  # 离散积分近似
            # 公式 11a：加速度
            x_a_dot2 = - part['m'][i]* p_star * Vab_squared * dw
             # 公式 11b：能量变化率
            D['e'][i] = - part['m'][i] * p_star * (u_star - part['u'][i]- 0.5 * dt * x_a_dot2) * Vab_squared * dw
            # 46a: 更新位置
            ipart['x'] += (part['u'] + 0.5 *dt * x_a_dot2 ) * dt
            # 46b: 更新速度
            ipart['u'] += x_a_dot2 * dt
            # 46c: 更新能量
            ipart['e'] += D['e'] * dt

            # 47a: 计算rho*，改变光滑长度
            rho_star = np.zeros_like(part['d'])
            C_smooth = 2
            y = 1
            h_scaled = C_smooth * part['h'][i]
            W_ab = Wf.W(2,xij, h_scaled)  # 计算核函数值
            rho_star[i] += part['m'][j] * W_ab
            # 47b: 更新平滑长度
            h_new = y * (part['m'] / rho_star) 
            # 47c: 更新密度
            W_ab = Wf.W(2, xij,h_new)
            ipart['d'] += part['m'][j] * W_ab

    return ipart, dt   #让ipart更新，part不变
