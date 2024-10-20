import numpy as np

def timestep(part, N, alpha, D):
    """
    计算SPH方法中的时间步长，该方法基于粒子的声速，这是由于声速在冲击波问题中的重要性。

    参数:
    part (dict): 粒子的属性，包含以下键:
                 - h: 平滑长度数组
                 - c: 声速数组
    N (int): 粒子总数
    alpha (array): SPH参数数组
    D (dict): 粒子的属性导数，包含以下键:
              - u: 速度导数组

    返回:
    float: 时间步长
    """
    
    dtc_list = []
    dti_list = []

    for i in range(N):
        try:
            # 检查 part['c'][i] 和 D['u'][i] 是否为标量
            if np.isscalar(part['c'][i]) and np.isscalar(D['u'][i]):
                # 计算基于声速的时间步长约束
                dtc = part['h'][i] / (abs(part['c'][i]) + 0.6 * alpha[0] * abs(part['c'][i]))
                dtc_list.append(dtc)
                
                # 计算基于速度导数的时间步长约束
                if abs(D['u'][i]) > 0:  # 防止除以零
                    dti = np.sqrt(part['h'][i] / abs(D['u'][i]))
                else:
                    dti = np.inf  # 如果速度导数为零，设定为无限大的时间步长
                dti_list.append(dti)
            else:
                raise ValueError(f'Particle {i}: c or u is not a scalar. c = {part["c"][i]}, u = {D["u"][i]}')

        except Exception as e:
            print(f"An unexpected error occurred for particle {i}: {e}")
            dtc_list.append(np.inf)  # 设置为一个较大的值以避免影响最小值选择
            dti_list.append(np.inf)  # 设置为一个较大的值以避免影响最小值选择

    # 获取最小的时间步长约束
    dtc_min = min(dtc_list)
    dti_min = min(dti_list)
    
    # 计算最终的时间步长
    dt = min(0.4 * dtc_min, 0.25 * dti_min)
    
    return dt
