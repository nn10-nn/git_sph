import numpy as np
from scipy.optimize import fsolve

# 已知参数
n_left = 640  # 左状态粒子数
rho_left = 1.0  # 左状态目标密度
x_left = np.linspace(0, 0.5, n_left)  # 粒子均匀分布, 生成640个等间距的点
m = 0.5 / (8 * 80)  # 每个粒子的质量
dx1 = 0.5 / (8 * 80)  # 左侧空间步长

# 高斯核函数  
def gaussian_kernel(r, h):
    return (1 / (h * np.sqrt(np.pi))) * np.exp(-(r / h)**2)

# 密度差函数，目标是通过光滑长度h得到目标密度rho_left
def density_difference(h, dx, n, m, rho_target):
    terms = [m * gaussian_kernel(k * dx, h) for k in range(1, n + 1)]  # 计算所有有效k的加权和
    rho_i = np.sum(terms)  # 计算粒子i的密度估计
    return rho_i - rho_target  # 返回与目标密度的差

# 求解光滑长度h，使得密度为目标密度rho_left
alpha_initial_guess = 0.5  # 初始猜测光滑长度的比例（可以调整）

# 使用fsolve求解h
h_solution = fsolve(density_difference, alpha_initial_guess, args=(dx1, n_left, m, rho_left))[0]
n =int (h_solution / dx1)
# 输出结果
print(f"n = {n}")
print(f"求解的光滑长度 h = {h_solution:.6f}")
