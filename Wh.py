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

# 密度差函数  先算左状态  左右状态分别对应不同的α
def density_difference(alpha, n, dx, m, rho_target):
    h = alpha * dx  # 光滑长度
    # r = n * dx   粒子间距离
    terms = np.array([np.exp(-(k / alpha)**2) for k in range(1, n + 1)])
    rho_i = m / (h * np.sqrt(np.pi)) * np.sum(terms)
    return rho_i - rho_target  # 返回密度差

# 求解α和h
for n in range(1, 21):  # 测试n = 1, 2, ..., 20
    alpha_initial_guess = n + 0.5  # α的初始猜测值
    # 使用fsolve求解α
    try:
        alpha_solution = fsolve(density_difference, alpha_initial_guess, args=(n, dx1, m, rho_left))[0]
        h_solution = alpha_solution * dx1  # 计算h
        # 输出结果
        print(f"n = {n}:")
        print(f"  求解的α: {alpha_solution}")
        print(f"  求解的h: {h_solution}")
        print()
    except RuntimeError:
        # 如果fsolve无法收敛，则打印一条错误消息并继续下一个n值
        print(f"n = {n}: fsolve无法收敛到解。")
        print()

