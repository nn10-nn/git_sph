import numpy as np

def DW(opt, r, h):
    """
    计算平滑核梯度。
    
    参数:
    opt (int): 控制使用的核类型。
               1 - cubic spline
               2 - gaussian
               3 - supergaussian
               4 - quartic
    r (float): 两个粒子之间距离的绝对值。
    h (float): 两个粒子之间平滑长度的平均值。
    sigma: 缩放因子
    返回:
    f (float): 平滑核梯度结果。
    """
    
    if opt == 1:  # CUBIC SPLINE KERNEL GRADIENT（三次样条核函数梯度）
        sigma = 1 / h
        s = r / h
        if s <= 1:
            f = -2 * s + 1.5 * s ** 2
        elif s <= 2.0:
            f = -0.5 * (2.0 - s) ** 2
        else:
            f = 0
    
    elif opt == 2:  # GAUSSIAN KERNEL GRADIENT9（高斯核梯度）
        sigma = 1 / (h * np.sqrt(np.pi))
        s = r / h
        if s <= 3:
            f = -2 * s * np.exp(-s ** 2)
        else:
            f = 0
    
    elif opt == 3:  # SUPERGAUSSIAN KERNEL GRADIENT（超高斯核梯度）
        sigma = 1 / (h * np.sqrt(np.pi))
        s = r / h
        if s <= 3:
            f = (-2 * s * (3 / 2 - s ** 2) * np.exp(-s ** 2)) - 2 * s * np.exp(-s ** 2)
        else:
            f = 0
    
    elif opt == 4:  # QUARTIC KERNEL GRADIENT（四次核梯度）
        sigma = 1 / h
        s = r / h
        if 0 <= s <= 2:
            f = - (9 / 4) * s + (19 / 8) * s ** 2 - (5 / 8) * s ** 3
        elif s > 2:
            f = 0
    
    return f * sigma