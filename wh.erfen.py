import numpy as np

# 高斯核函数  
def gaussian_kernel(r, h):
    return (1 / (h * np.sqrt(np.pi))) * np.exp(-(r / h)**2)

# 计算质量密度估计函数
def calculate_density(dx, m, n, h, rho_target):
    rho_i = 0
    for k in range(1, n + 1):
        r = k * dx  # 粒子之间的距离
        rho_i += m * gaussian_kernel(r, h)
    return rho_i

# 二分法求解光滑长度h
def bisection_method(dx, m, n, rho_target, a, b, tol=1e-6, max_iter=500):
    """
    tol (float): 容忍误差（默认1e-6）
    max_iter (int): 最大迭代次数（默认500）
    """
    iter_count = 0 #迭代次数
    while iter_count < max_iter:
        mid = (a + b) / 2
        rho_i = calculate_density(dx, m, n, mid, rho_target)

        # 输出每次迭代的结果，帮助调试
        print(f"迭代次数：{iter_count+1}, 中点h = {mid:.6f}, 计算的rho_i = {rho_i:.6f}, 目标密度 = {rho_target}")

        if abs(rho_i - rho_target) < tol:
            print(f"求解成功：h = {mid}, 计算的密度 = {rho_i}")
            return mid
        elif rho_i > rho_target:
            b = mid
        else:
            a = mid
        
        iter_count += 1
    
    print("达到最大迭代次数")
    return (a + b) / 2  # 返回中点

def find_initial_bracket(dx, m, n, rho_target, h_min=1e-5, h_max=1.0, step_factor=1.5, max_trials=500):
    """
    寻找满足二分法条件的初始区间 [a,b]
    参数：
        dx, m, n, rho_target: 与calculate_density()一致
        h_min, h_max: h的初始范围
        step_factor: 每次扩大范围的倍数（例如1.5）
        max_trials: 最大搜索次数
    返回：
        满足条件的(a,b)
    """
    a, b = h_min, h_max
    rho_a = calculate_density(dx, m, n, a, rho_target)
    rho_b = calculate_density(dx, m, n, b, rho_target)

    trial = 0
    while (rho_a - rho_target) * (rho_b - rho_target) > 0 and trial < max_trials:
        # 扩大搜索范围
        a /= step_factor
        b *= step_factor
        rho_a = calculate_density(dx, m, n, a, rho_target)
        rho_b = calculate_density(dx, m, n, b, rho_target)
        trial += 1

    if trial == max_trials:
        raise ValueError("未找到合适的初始区间，请检查参数或密度目标的合理性。")
    
    print(f"自动搜索得到的初始区间：[a={a:.6f}, rho(a)={rho_a:.6f}], [b={b:.6f}, rho(b)={rho_b:.6f}]")
    return a, b


# 左状态参数
n_left = 640  # 左状态粒子数
rho_left = 1.0  # 左状态目标密度
dx1 = 0.5 / (8 * 80)  # 左侧空间步长
m = 0.5 / (8 * 80)  # 每个粒子的质量

# 右状态参数
n_right = 80  # 右状态粒子数
rho_right = 0.125  # 右状态目标密度
dx2 = 0.5 / 80  # 右侧空间步长

# 使用二分法求解左状态的光滑长度
print("===== 左状态求解 =====")
a_left = 0.001  # 左状态光滑长度初始区间左边界
b_left = 0.1  # 左状态光滑长度初始区间右边界

a_left, b_left = find_initial_bracket(dx1, m, n_left, rho_left)
h_solution_left = bisection_method(dx1, m, n_left, rho_left, a_left, b_left)

# 验证左状态的质量密度估计
rho_i_left = calculate_density(dx1, m, n_left, h_solution_left, rho_left)
if abs(rho_i_left - rho_left) < 1e-6:
    print(f"  左状态验证通过：计算的rho_i接近目标密度（rho_left={rho_left}）")
else:
    print(f"  左状态验证失败：计算的rho_i = {rho_i_left}, 目标密度 = {rho_left}")

# 使用二分法求解右状态的光滑长度
print("\n===== 右状态求解 =====")
a_right = 0.001  # 右状态光滑长度初始区间左边界
b_right = 0.1  # 右状态光滑长度初始区间右边界
a_right, b_right = find_initial_bracket(dx2, m, n_right, rho_right)
h_solution_right = bisection_method(dx2, m, n_right, rho_right, a_right, b_right)

# 验证右状态的质量密度估计
rho_i_right = calculate_density(dx2, m, n_right, h_solution_right, rho_right)
if abs(rho_i_right - rho_right) < 1e-6:
    print(f"  右状态验证通过：计算的rho_i接近目标密度（rho_right={rho_right}）")
else:
    print(f"  右状态验证失败：计算的rho_i = {rho_i_right}, 目标密度 = {rho_right}")
