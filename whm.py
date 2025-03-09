import numpy as np
import math

# 参数设置
n_left = 640  # 左状态粒子数
rho_left = 1.0  # 左状态目标密度
n_right = 80  # 右状态粒子数
rho_right = 0.125  # 右状态目标密度

# 粒子分布和步长
dx_left = 0.5 / (8 * 80)  # 左侧空间步长
dx_right = 0.5 / 80  # 右侧空间步长

# 高斯核函数
def gaussian_kernel(r, h):
    return (1 / (h * np.sqrt(np.pi))) * np.exp(-(r / h)**2)

# 计算密度差
def density_difference(alpha, dx, m, rho_target):
    h = alpha * dx  # 光滑长度等于dx * alpha
    max_k = math.floor(alpha)  # 最大有效k值（r=kdx，k是比α小的最大整数）
    valid_ks = range(1, max_k + 1)  # 仅考虑有效的k值，确保不会超出粒子数量
    terms = [gaussian_kernel(k * dx, h) for k in valid_ks]  # 计算每个k对应的高斯核值
    rho_i = m * np.sum(terms)  # 计算总的粒子密度
    return rho_i - rho_target  # 返回密度差

# 计算每个粒子的质量
def calculate_mass(alpha, dx, n, rho_target):
    m_guess = 1.0  # 先假设每个粒子的质量为1
    # 计算密度差，目的是找到使得密度差接近0的质量
    density_diff = density_difference(alpha, dx, m_guess, rho_target)
    # 通过目标密度和估算密度差来调整质量，找到与目标密度匹配的质量
    # 这里我们使用简单的线性比例来调整质量
    m_adjusted = m_guess * (rho_target / (rho_target + density_diff))
    return m_adjusted

# 给定的alpha值
alpha_left = 8  # 左状态的alpha
alpha_right = 2  # 右状态的alpha

# 计算左右状态的质量
mass_left = calculate_mass(alpha_left, dx_left, n_left, rho_left)
mass_right = calculate_mass(alpha_right, dx_right, n_right, rho_right)

# 输出左右状态的质量
print(f"左状态的质量: {mass_left:.6f}")
print(f"右状态的质量: {mass_right:.6f}")
