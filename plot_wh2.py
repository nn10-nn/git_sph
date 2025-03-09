import numpy as np
import matplotlib.pyplot as plt
import math

# 粒子参数（左状态和右状态）
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
m_left = 0.5 / (8 * 80 )  # 左状态粒子质量
m_right = 0.5 / (8 )  # 右状态粒子质量

# alpha的取值范围
alpha_values = np.linspace(1.1, 11.9, 100)  # 生成100个alpha值，范围从1.1到11.9，避免端点问题

# 计算左状态的density_difference
density_differences_left = [density_difference(alpha, dx_left, m_left, rho_left) for alpha in alpha_values]

# 计算右状态的density_difference
density_differences_right = [density_difference(alpha, dx_right, m_right, rho_right) for alpha in alpha_values]

# 绘制并保存左状态的图像
plt.plot(alpha_values, density_differences_left, label='Left State', color='b')
plt.xlabel('alpha')
plt.ylabel('density_difference (Left State)')
plt.title('Density Difference vs. Alpha (Left State)')
plt.grid(True)
plt.legend()
plt.savefig("density_difference_left.png")  # 保存左状态图像
plt.show()  # 显示左状态图像

# 绘制并保存右状态的图像
plt.plot(alpha_values, density_differences_right, label='Right State', color='r')
plt.xlabel('alpha')
plt.ylabel('density_difference (Right State)')
plt.title('Density Difference vs. Alpha (Right State)')
plt.grid(True)
plt.legend()
plt.savefig("density_difference_right.png")  # 保存右状态图像
plt.show()  # 显示右状态图像
