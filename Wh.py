import numpy as np
import Wf
from scipy.optimize import fsolve

# 左状态参数
n_left = 640  # 左状态粒子数
rho_left = 1.0  # 左状态密度
x_left = np.linspace(0, 0.5, n_left)  # 粒子均匀分布,所以在区间中生成640个等间距的点
m = 0.5 / (8 * 80)  # 每个粒子的质量
dx1 =  0.5 / (8 * 80)           # 左侧空间步长  这里是8,一共有9n个粒子，所以密度比0.1和0.125是8比1
# 右状态参数
n_right = 80  # 右状态粒子数
rho_right = 0.125  # 右状态密度
x_right = np.linspace(0.5, 1.0, n_right)  # 右状态粒子均匀分布，在区间中生成80个等间距的点
m = 0.5 / (8 * 80)  # 每个粒子的质量
dx2 = 0.5 / 80                   # 右侧空间步长

# 高斯核函数
def gaussian_kernel(r, h):
    return (1 / (h * np.sqrt(np.pi))) * np.exp(-(r**2) / h**2)

# 密度差函数  先算左状态  左右状态分别对应不同的α
def density_difference(alpha, n, dx, m, rho_target):
    h = alpha * dx1  # 光滑长度
    r = n * dx1  # 粒子间距离
    W = gaussian_kernel(r, h)  # 核函数值
    #rho_i =  np.sum( m * W )  # 密度估计  
    #n=1时
    #rho_i = m * 1 / (h * np.sqrt(np.pi)) * np.exp(-(1/alpha)**2) 
    #n=3
    #rho_i = m * 1 / (h * np.sqrt(np.pi)) * np.exp(-(1/alpha)**2+np.exp(-(2/alpha)**2) )
    #n=3时 
    rho_i = m * 1 / (h * np.sqrt(np.pi)) *np.exp(-(1/alpha)**2+np.exp(-(2/alpha)**2)+np.exp(-(3/alpha)**2))
    return rho_i - rho_target  # 返回密度差

n=3

# 求解 α 和 h
#for n in range(1, 20):  # 测试 n = 1, 2, 3, ……
    #n = 1
    # 初始猜测值
alpha_initial_guess = n + 0.5  # α 的初始猜测值（0.5 是一个折中的值，既不太接近n也不太接近n+1,有助于 fsolve 更快地收敛到真实解）

    # 使用 fsolve 求解 α
alpha_solution = fsolve(density_difference, alpha_initial_guess, args=(n, dx1, m, rho_left))
h_solution = alpha_solution * dx1  # 计算 h

    # 输出结果
print(f"n = {n}:")
print(f"  求解的 α: {alpha_solution[0]}")
print(f"  求解的 h: {h_solution[0]}")
print()
#计算a的平方