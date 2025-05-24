import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# calculate initial angle
# import math
# print (math.atan(1.1206635236740112 -0.8241485953330994)/(2.72072696685791-2.9562156200408936))

def getTraj(motor_v):

    gear_ratio = 1/1.2
    flywheel_r = 0.05
    
    efficiency = 0.96

    m = 0.270    # 球体质量 (kg)
    g = 9.8    # 重力加速度 (m/s²)
    r = 0.12    # 球体半径 (m)
    A = np.pi * r**2  # 投影面积 (m²)
    rho = 1.225  # 空气密度 (kg/m³)
    Cd = 0.7    # 阻力系数
    Cl = 0.2    # 升力系数
    v0 = efficiency * (motor_v/gear_ratio) * 2 * np.pi * flywheel_r / 2 # 初始速度 (m/s)
    # print(v0)
    omega = v0 / r  # 角速度 (rad/s)

    theta = 0.96
    # 发射角度 (转换为弧度)

    # 初始速度分量
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)

    # 初始状态 [x, y, vx, vy]
    initial_state = [0.0, 0.0, vx0, vy0]

    def projectile_motion(t, state):
        x, y, vx, vy = state
        v = np.sqrt(vx**2 + vy**2)
        
        # 空气阻力计算
        Fd = 0.5 * rho * Cd * A * v**2
        Fd_x = -Fd * vx/v if v != 0 else 0
        Fd_y = -Fd * vy/v if v != 0 else 0
        
        # Magnus力计算
        FM_coeff = 0.5 * rho * A * r * Cl
        FM_x = FM_coeff * (-omega * vy)
        FM_y = FM_coeff * (omega * vx)
        
        # 加速度计算
        ax = (Fd_x + FM_x) / m
        ay = (Fd_y + FM_y)/m - g
        
        return [vx, vy, ax, ay]

    # 接地事件检测
    def hit_ground(t, state):
        return state[1]
    hit_ground.terminal = True
    hit_ground.direction = -1

    # 数值积分
    sol = solve_ivp(projectile_motion, 
                    t_span=(0, 10), 
                    y0=initial_state, 
                    events=hit_ground,
                    max_step=0.01)

    # 提取结果
    x = sol.y[0]
    y = sol.y[1]
    return x, y 
if __name__ == "__main__":
    x, y = getTraj(60)
    # 绘制轨迹
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='With Magnus Effect')
    plt.title('Projectile Trajectory with Air Drag and Magnus Effect')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Height (m)')
    plt.grid(True)
    plt.legend()
    plt.show()