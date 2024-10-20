import numpy as np
import matplotlib.pyplot as plt

def plotSodTubeResults(part, data, makeVideo, k=None):
    """
    绘制数值分析结果，应用于Sod管的SPH方法。

    参数:
    part (dict): 包含粒子属性的字典，包含以下键:
                 - x: 粒子位置数组
                 - d: 粒子密度数组
                 - p: 粒子压力数组
                 - u: 粒子速度数组
                 - e: 粒子内部能量数组
    data (dict): 包含参考数据的字典，包含以下键:
                 - x: 参考数据位置数组
                 - rho: 参考数据密度数组
                 - P: 参考数据压力数组
                 - u: 参考数据速度数组
                 - e: 参考数据内部能量数组
    makeVideo (bool): 是否生成视频
    k (int, optional): 当前帧的索引，用于生成视频文件名
    """

    plt.figure(figsize=(7, 7))
    
    # 密度图
    plt.subplot(2, 2, 1)
    plt.plot(part['x'], part['d'], 'ok', label='SPH Data', markersize=1.2)
    plt.plot(data['x'], data['rho'], 'k-', label='Reference Data', linewidth=0.75)
    plt.xlabel('Position (m)')
    plt.ylabel('Density (kg/m^3)')
    plt.xlim((-0.6, 0.6))
    plt.ylim((0.15, 1.1))
    plt.legend()
 
    
    # 压力图
    plt.subplot(2, 2, 2)
    plt.plot(part['x'], part['p'], 'ok', markersize=1.2)
    plt.plot(data['x'], data['P'], 'k-', linewidth=0.75)
    plt.xlabel('Position (m)')
    plt.ylabel('Pressure (kPa)')
    plt.axis([-0.6, 0.6, 0.1, 1.1])
    
    # 速度图
    plt.subplot(2, 2, 3)
    plt.plot(part['x'], part['u'], 'ok', markersize=1.2)
    plt.plot(data['x'], data['u'], 'k-', linewidth=0.75)
    plt.xlabel('Position (m)')
    plt.ylabel('Velocity (m/s)')
    plt.axis([-0.6, 0.6, -0.05, 0.9])
    
    # 内能图
    plt.subplot(2, 2, 4)
    plt.plot(part['x'], part['e'], 'ok', markersize=1.2)
    plt.plot(data['x'], data['e'], 'k-', linewidth=0.75)
    plt.xlabel('Position (m)')
    plt.ylabel('Internal Energy (kJ/kg)')
    plt.axis([-0.6, 0.6, 1.7, 2.7])
    

    plt.show()
    
  
