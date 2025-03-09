import numpy as np
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
dx2 = 0.5 / 80                   # 右侧空间步长

# 高斯核函数  
def gaussian_kernel(r, h):
    return (1 / (h * np.sqrt(np.pi))) * np.exp(-(r / h)**2)

# 密度差函数
def density_difference(alpha, dx, n, m, rho_target):
    h = alpha * dx
    max_k = int(h / dx)  # 最大有效k值（r=kdx,k是比α小的最大整数）
    valid_ks = range(1, min(n, max_k) + 1)  # 仅考虑存在的粒子，确保不会考虑超过实际粒子数量的k值
    terms = [ gaussian_kernel(k*dx, h) for k in valid_ks]  #
    rho_i = m * np.sum(terms)
    return rho_i - rho_target


# ------------------------- 求解左状态 -------------------------
print("===== 左状态求解 =====")
for n in range(1, 11):  # 测试n=1到10
    alpha_initial_guess = n + 0.5  # 初始猜测值
    try:
        # 调用fsolve求解alpha
        alpha_solution = fsolve(
            density_difference, 
            alpha_initial_guess, 
            args=(dx1, n, m, rho_left)
        )[0]
        h_solution = alpha_solution * dx1  # 计算h
        # 输出结果
        print(f"n={n}:")
        print(f"  求解的α = {alpha_solution:.3f}")
        print(f"  求解的h = {h_solution:.6f}")
        print(f"  α是否在({n}, {n+1})区间内？{'是' if n < alpha_solution < n+1 else '否'}")
        print()
    except Exception as e:
        print(f"n={n}: 求解失败，错误信息：{str(e)}")
        print()
# ------------------------- 求解右状态 -------------------------
print("\n===== 右状态求解 =====")
for n in range(1, 11):  # 测试n=1到10
    alpha_initial_guess = n + 0.5  # 初始猜测值
    try:
        # 调用fsolve求解alpha
        alpha_solution = fsolve(
            density_difference, 
            alpha_initial_guess, 
            args=(dx2, n, m, rho_right)  # 注意这里传入右状态参数：dx2和rho_right
        )[0]
        h_solution = alpha_solution * dx2  # 计算h
        # 输出结果
        print(f"n={n}:")
        print(f"  求解的α = {alpha_solution:.3f}")
        print(f"  求解的h = {h_solution:.6f}")
        print(f"  α是否在({n}, {n+1})区间内？{'是' if n < alpha_solution < n+1 else '否'}")
        print()
    except Exception as e:
        print(f"n={n}: 求解失败，错误信息：{str(e)}")
        print()