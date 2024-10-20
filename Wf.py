import numpy as np

def W(opt, r, h):
    """
    计算平滑粒子流体动力学(SPH)平滑核函数的梯度。

    参数:
    opt (int): 选择平滑核的类型
               1 - 三次样条平滑核
               2 - 高斯平滑核
               3 - 超高斯平滑核
               4 - 四次平滑核
    r (float): 两个粒子之间的距离绝对值
    h (float): 平滑长度的平均值

    返回:
    float: 平滑核函数的梯度结果
    """
    
    if opt == 1:
        # 三次样条平滑核 (Cubic Spline Kernel)
        sigma = 1 / h
        s = r / h
        if s <= 1:
            # 核函数在距离s小于等于1时的计算公式
            f = 2/3 - s**2 + 0.5*s**3
        elif s <= 2:
            # 核函数在距离s大于1且小于等于2时的计算公式
            f = ((2 - s)**3) / 6
        else:
            # 核函数在距离s大于2时为0
            f = 0

    elif opt == 2:
        # 高斯平滑核 (Gaussian Kernel)
        sigma = 1 / (h * np.sqrt(np.pi))
        s = r / h
        if s <= 3:
            # 核函数在距离s小于等于3时的计算公式
            f = np.exp(-s**2)
        else:
            # 核函数在距离s大于3时为0
            f = 0

    elif opt == 3:
        # 超高斯平滑核 (Supergaussian Kernel)
        sigma = 1 / (h * np.sqrt(np.pi))
        s = r / h
        if s <= 3.0:
            # 核函数在距离s小于等于3时的计算公式，带有一个额外的乘数因子(3/2 - s^2)
            f = np.exp(-s**2) * (3/2 - s**2)
        else:
            # 核函数在距离s大于3时为0
            f = 0

    elif opt == 4:
        # 四次平滑核 (Quartic Kernel)
        sigma = 1 / h
        s = r / h
        if 0 <= s <= 2:
            # 核函数在距离s在0到2之间时的计算公式
            f = ((2/3) - (9/8)*s**2 + (19/24)*s**3 - (5/32)*s**4) 
        else:
            # 核函数在距离s大于2时为0
            f = 0

    # 乘以sigma以得到最终结果
    return f * sigma
