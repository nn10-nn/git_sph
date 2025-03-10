import numpy as np 
import math

# 左状态参数
n_left = 640  # 左状态粒子数
rho_left = 1.0  # 左状态密度
x_left = np.linspace(0, 0.5, n_left)  # 粒子均匀分布, 所以在区间中生成640个等间距的点
m = 0.0020  # 每个粒子的质量
dx1 = 0.5 / (8 * 80)  # 左侧空间步长 这里是8，一共有9n个粒子，所以密度比0.1和0.125是8比1

# 右状态参数
n_right = 80  # 右状态粒子数
rho_right = 0.125  # 右状态密度
x_right = np.linspace(0.5, 1.0, n_right)
dx2 = 0.5 / 80  # 右侧空间步长

# 高斯核函数  
def gaussian_kernel(r, h):
    return (1 / (h * np.sqrt(np.pi))) * np.exp(-(r / h)**2)

# 计算质量密度估计函数
def calculate_density(alpha, dx, m, n, h, rho_target):
    h = alpha * dx
    max_k = math.floor(alpha)  # 最大有效k值（r=kdx,k是比α小的最大整数）
    valid_ks = range(1, min(n, max_k)   + 1)  # 仅考虑存在的粒子，确保不会考虑超过实际粒子数量的k值
    terms = [gaussian_kernel(k * dx, h) for k in valid_ks]
    rho_i = m * np.sum(terms)
    return rho_i

# ------------------------- 求解左状态 -------------------------
print("===== 左状态求解 =====")
rho_i_left = calculate_density( 6.002,dx1, m, 6, 0.004689, rho_left)
print(f"  左状态密度：计算的rho_i为（rho_left={rho_left}）")

# ------------------------- 求解右状态 -------------------------
print("\n===== 右状态求解 =====")

rho_i_right = calculate_density(6.002,dx2, m, 6, 0.037512, rho_right)
print(f"  右状态密度：计算的rho_i为（rho_right={rho_right}）")


