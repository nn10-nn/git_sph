import numpy as np

def preProcess(N, dx1, dx2, gamma):
    """
    初始化一维Sod管中每个粒子的属性。

    参数:
    N (int): 粒子总数
    dx1 (float): 左侧空间步长
    dx2 (float): 右侧空间步长
    gamma (float): 气体常数

    返回:
    dict: 包含粒子属性的字典，包括位置、密度、压力、速度、内部能量、声速、质量和平滑长度
    """
    
    # 初始化粒子属性字典，将所有属性初始化为一维数组
    part = {'x': np.zeros(N),  # 粒子位置
            'd': np.zeros(N),  # 粒子密度
            'p': np.zeros(N),  # 粒子压力
            'u': np.zeros(N),  # 粒子速度
            'e': np.zeros(N),  # 粒子内部能量
            'c': np.zeros(N),  # 声速
            'm': np.zeros(N),  # 粒子质量
            'h': np.zeros(N)}  # 平滑长度
    gamma = 1.4
    part['gamma'] = gamma
 

    for i in range(N):
        if i < ((N / 5) * 4 + 1):
            # 左侧粒子的位置计算
            # 非均匀分布的粒子属性
            part['x'][i] = -(0.5 - dx1/2.0) + dx1 * i
            # 均匀分布的粒子属性
            part['x'][i] = -0.5 + dx1 * i
            part['d'][i] = 1.0
            part['e'][i] = gamma 
            part['p'][i] = 1  # 计算压力
            part['u'][i] = 0.0  # 初始速度设置为0
            part['c'][i] = np.sqrt((gamma  * (gamma-1) )* part['e'][i])  # 计算声速
            part['m'][i] = dx1  # 用密度part['d'][i]质量密度估计公式算出mb
            part['h'][i] = 6.002 * dx1  # ，利用m即dx1和rho:1.0去求先求W，再用W求h平滑长度

        else:
            # 右侧粒子的位置计算
            # 非均匀分布的粒子属性
            part['x'][i] = dx2*(i-((N/5.0)*4 + 1) + 0.5)
            # 均匀分布的粒子属性
            part['x'][i] = 0.0 + dx2*(i - (N/5.0)*4.0)
            part['d'][i] = 0.125
            part['e'][i] = gamma * 0.1 / 0.125 
            part['p'][i] = 0.1  # 计算压力
            part['u'][i] = 0.0  # 初始速度设置为0
            part['c'][i] = np.sqrt((gamma  * (gamma-1) )* part['e'][i])  # 计算声速
            part['m'][i] = dx1  # 用密度part['d'][i]质量密度估计公式算出mb
            part['h'][i] = 6.002 * dx2  # 平滑长度  重新算2.0

    return part
