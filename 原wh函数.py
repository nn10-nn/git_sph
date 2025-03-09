import numpy as np
import Wf
from scipy.optimize import fsolve

def density_equation(h):   #直接输入rho_target ，h=a*dx(a是小数)，|x|=n*dx, (n是比α小的最大的整数值)
    rho_target = 1.0 #左侧初始密度
    dx1 = 0.5 / (8 * n)  =   0.5 / (8 * 80)  
    h = a * dx1
    #rho_target = 0.125 右侧初始密度
    r_ij = np.abs(x_i - x_j)  # 计算距离
    # 计算核函数值:
    sigma = 1 / (h * np.sqrt(np.pi))
    s = r / h
    if s <= 3.0:
    # 核函数在距离s小于等于3时的计算公式
        f = np.exp(-s**2)
    else:
            # 核函数在距离s大于3时为0
        f = 0
 
    W_values = f * sigma  # 计算核函数值
    rho_i = np.sum(m_j * W_values)  # 计算估计密度，或者换成符号导数运算，改成for循环
    return rho_i - rho_target  # 返回密度差
#不调用函数Wf，直接写在Wh函数里

# 左状态参数
n_left = 320  # 左状态粒子数
rho_left = 1.0  # 左状态密度
x_left = np.linspace(0, 0.5, n_left)  # 粒子均匀分布,所以在区间中生成320个等间距的点
m_left = 0.5 / (8 * 80)  # 每个粒子的质量

# 右状态参数
n_right = 80  # 右状态粒子数
rho_right = 0.125  # 右状态密度
x_right = np.linspace(0.5, 1.0, n_right)  # 右状态粒子均匀分布，在区间中生成80个等间距的点
m_right = 0.5 / (8 * 80)  # 每个粒子的质量

# 初始猜测的光滑长度,fsolve 从这个值开始迭代，逐步逼近方程的解:density_equation = 0
initial_guess = 0.01

# 计算左状态的光滑长度 h1
x_i_left = 0.25  # 左状态选择一个粒子的位置
h1_solution = fsolve(density_equation, initial_guess, args=(x_i_left, x_left, m_left, rho_left))

print(x_i_left)

# 计算右状态的光滑长度 h2
x_i_right = 0.75  # 右状态选择一个粒子的位置
h2_solution = fsolve(density_equation, initial_guess, args=(x_i_right, x_right, m_right, rho_right))

print(f"左状态的光滑长度 h1 = {h1_solution[0]:.6f}")
print(f"右状态的光滑长度 h2 = {h2_solution[0]:.6f}")
