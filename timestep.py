import numpy as np

def timestep(part, N):
    """
    计算SPH方法中的时间步长，该方法基于粒子的声速，这是由于声速在冲击波问题中的重要性。

    参数:
    part (dict): 粒子的属性，包含以下键:
                 - h: 平滑长度数组
                 - c: 声速数组
    N (int): 粒子总数

    返回:
    float: 时间步长
    """
    
    dtc_list = []

    for i in range(N):
        try:
            # 确保声速是标量且非零，以防止无效的计算
            if np.isscalar(part['c'][i]) and part['c'][i] != 0:
                # 基于声速和平滑长度计算时间步长约束
                dtc = part['h'][i] / abs(part['c'][i])
                dtc_list.append(dtc)

        except Exception as e:
            print(f"An unexpected error occurred for particle {i}: {e}")
            dtc_list.append(np.inf)  # 设置为一个较大的值以避免影响最小值选择

    # 获取最小的时间步长约束
    dtc_min = min(dtc_list)

    
    # 计算最终的时间步长  0.25是CFL数
    dt = 0.25 * dtc_min
    
    return dt
