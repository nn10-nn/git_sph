import numpy as np

def integration(opt, part, ipart, N, D, dt):
    """
    使用二阶龙格-库塔方法对粒子的属性进行时间积分。

    参数:
    opt (int): 控制使用哪一步积分方法（1 代表中间步骤，其他代表最终步骤）。
    part (dict): 包含粒子当前属性的字典。
    ipart (dict): 包含粒子中间属性的字典。
    N (int): 粒子总数。
    D (dict): 包含粒子属性导数的字典。
    dt (float): 时间步长。

    返回:
    tuple: 更新后的速度、内能和位置数组。
    """
    u = np.zeros(N)
    e = np.zeros(N)
    x = np.zeros(N)

    if opt == 1:  # 第一步的二阶龙格-库塔
        for i in range(N):
            u[i] = ipart['u'][i] + dt * D['u'][i] / 2  # 计算 t + dt/2 的速度
            e[i] = ipart['e'][i] + dt * D['e'][i] / 2  # 计算 t + dt/2 的内能
            x[i] = ipart['x'][i] + dt * D['x'][i] / 2  # 计算 t + dt/2 的位置
    else:  # 第二步的二阶龙格-库塔
        for i in range(N):
            u[i] = (ipart['u'][i] + part['u'][i] + dt * D['u'][i]) / 2  # 计算 t + dt 的速度
            e[i] = (ipart['e'][i] + part['e'][i] + dt * D['e'][i]) / 2  # 计算 t + dt 的内能
            x[i] = (ipart['x'][i] + part['x'][i] + dt * D['x'][i]) / 2  # 计算 t + dt 的位置

    return u, e, x
