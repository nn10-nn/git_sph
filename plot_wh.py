import numpy as np 
import matplotlib.pyplot as plt
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
x_right = np.linspace(0.5, 1.0, n_right)  # 右状态粒子均匀分布，在区间中生成80个等间距的点
dx2 = 0.5 / 80  # 右侧空间步长

# 高斯核函数  
def gaussian_kernel(r, h):
    return (1 / (h * np.sqrt(np.pi))) * np.exp(-(r / h)**2)

# 密度差函数
def density_difference(alpha, dx, m, rho_target):
    h = alpha * dx # 
    max_k = math.floor(alpha)  # 最大有效k值（r=kdx,k是比α小的最大整数）
    valid_ks = range(1, max_k +1)  # 仅考虑存在的粒子，确保不会考虑超过实际粒子数量的k值
    terms = [gaussian_kernel(k * dx, h) for k in valid_ks]
    rho_i = m * np.sum(terms)
    return rho_i - rho_target

alpha_values = np.linspace(1.1, 11.9, 100)  # 使用1.1到11.9以避免端点问题，并生成100个点

# 计算对应的density_difference值
density_differences1 = [density_difference(alpha, dx1, m, rho_left) for alpha in alpha_values]
density_differences2 = [density_difference(alpha, dx2, m, rho_right) for alpha in alpha_values]

# 绘图
plt.plot(alpha_values, density_differences1)
plt.xlabel('alpha')
plt.ylabel('density_difference1')
plt.title('Density Difference vs. Alpha')
plt.grid(True)
plt.savefig("density_difference_plot1.png") 

plt.plot(alpha_values, density_differences2)
plt.xlabel('alpha')
plt.ylabel('density_difference2')
plt.title('Density Difference vs. Alpha')
plt.grid(True)
plt.savefig("density_difference_plot2.png")  
plt.show()
