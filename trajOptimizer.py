import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pandas as pd

# 已知常量参数
m = 0.27       # 质量 (kg)
g = 9.8       # 重力加速度 (m/s²)
r = 0.12       # 球体半径 (m)
A = np.pi*r**2 # 投影面积 (m²)
rho = 1.225    # 空气密度 (kg/m³)
theta = 0.98  # 发射角度 (固定)

speed = 60
index = 2

def readCoordinates(speed, index):
    data = pd.read_csv(f'dataset\\{speed}-{index}annotations.csv')
    x = data["x"].values
    y = data["y"].values
    x, y = y, x
    
    y *= -1
    
    x -= x[0]
    y -= y[0]
    
    # print("x:", x)
    # print("y:", y)
    return x, y

x_data, y_data = readCoordinates(speed, index)
    

def projectile_system(params, t_eval):
    """模拟抛射体运动并返回轨迹"""
    Cd, Cl, v0 = params
    omega = -v0 / r  # 根据优化参数计算旋转速度
    
    # 初始条件
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    initial_state = [0, 0, vx0, vy0]

    # 定义微分方程
    def deriv(t, state):
        x, y, vx, vy = state
        v = np.hypot(vx, vy)
        
        # 空气阻力
        Fd = 0.5 * rho * Cd * A * v**2
        Fd_x = -Fd * vx/v if v != 0 else 0
        Fd_y = -Fd * vy/v if v != 0 else 0
        
        # Magnus力
        FM = 0.5 * rho * A * r * Cl * omega
        FM_x = FM * (-vy)  # 二维旋转的叉积简化
        FM_y = FM * vx
        
        # 加速度
        ax = (Fd_x + FM_x) / m
        ay = (Fd_y + FM_y)/m - g
        
        return [vx, vy, ax, ay]

    # 接地事件检测
    def hit_ground(t, state):
        return state[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    # 数值求解
    sol = solve_ivp(deriv, 
                   t_span=[0, max(t_eval)+1],
                   y0=initial_state,
                   t_eval=t_eval,
                   events=hit_ground,
                   method='RK45',
                   atol=1e-9,
                   rtol=1e-6)
    
    return sol

def objective(params):
    """目标函数：计算模拟与实验数据的残差"""
    # 生成时间估计（根据抛物线运动特性自动生成）
    t_estimate = np.linspace(0, np.sqrt(2*max(y_data)/g)*2, len(x_data))
    
    # 运行模拟
    sol = projectile_system(params, t_estimate)
    
    # 处理提前落地的情况
    if len(sol.t) == 0:
        return np.full_like(x_data, 1e6)
    
    # 插值到实验数据点
    x_sim = np.interp(t_estimate, sol.t, sol.y[0])
    y_sim = np.interp(t_estimate, sol.t, sol.y[1])
    
    # 计算组合残差
    return np.concatenate([x_sim - x_data, y_sim - y_data])

# 初始参数猜测 [Cd, Cl, v0]
params0 = [4.7, 0.1, 25.0]

# 设置参数边界 (Cd, Cl, v0)
bounds = ([0.4, 0.1, 0], [5.0, 2.0, 50.0])

# 执行优化
result = least_squares(objective, 
                      params0, 
                      bounds=bounds,
                      loss='soft_l1',
                      f_scale=0.1,
                      verbose=2)

# 提取优化结果
Cd_opt, Cl_opt, v0_opt = result.x

# 最终模拟
t_final = np.linspace(0, np.sqrt(2*max(y_data)/g)*2, 200)
sol_final = projectile_system(result.x, t_final)

# 可视化结果
plt.figure(figsize=(12, 8))
plt.plot(x_data, y_data, 'ro', markersize=8, label='Experimental Data')
plt.plot(sol_final.y[0], sol_final.y[1], 
        'b-', linewidth=2, label=f'Optimized: Cd={Cd_opt:.3f}, Cl={Cl_opt:.3f}, v0={v0_opt:.1f}m/s')
plt.title(f'Projectile Motion Parameter Optimization speed:{speed}, index{index}', fontsize=14)
plt.xlabel('Horizontal Distance (m)', fontsize=12)
plt.ylabel('Vertical Height (m)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# 打印优化结果
print("\nOptimization Results:")
print(f"Drag Coefficient (Cd): {Cd_opt:.4f}")
print(f"Lift Coefficient (Cl): {Cl_opt:.4f}")
print(f"Initial Velocity (v0): {v0_opt:.2f} m/s")
print(f"Rotation Speed (ω): {v0_opt/r:.2f} rad/s")