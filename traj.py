import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


# calculate initial angle
# import math
# print (math.atan(1.1206635236740112 -0.8241485953330994)/(2.72072696685791-2.9562156200408936))

def getTraj(v0, theta_degree = 55):

    gear_ratio = 1
    flywheel_r = 0.05
    
    efficiency = 0.8

    g = 9.8    # 重力加速度 (m/s²)
    r = 3*0.0254    # 球体半径 (m)
    A = np.pi * r**2  # 投影面积 (m²)
    
    # v0 = efficiency * (motor_v/gear_ratio) * 2 * np.pi * flywheel_r * 0.75 # 初始速度 (m/s)
    
    
    # 2026 FRC "FUEL" Game Piece Specifications
    m = 0.215         # Mass (kg) - Official range is 0.448 - 0.5 lbs
    r = 0.075         # Radius (m) - Official diameter is 5.91 inches (150 mm)
    A = np.pi * r**2  # Projected Area (~0.0177 m²)
    
    # Aerodynamic Coefficients (Estimates for 5.91" foam sphere)
    # Cd for a sphere is typically 0.47. For textured foam, 0.45 - 0.50 is common.
    Cd = 0.2         
    
    # Cl (Lift) depends on the spin (Magnus effect). 
    # If your shooter uses a single flywheel, use ~0.1 to 0.2. 
    # If you have a top/bottom dual-roller with no net spin, set Cl = 0.
    Cl = 0.3        

    # Environmental Constants (remain the same)
    g = 9.81          # Gravity (m/s²)
    rho = 1.225       # Air density (kg/m³) at sea level
    
    
    # print(v0)
    omega = 0.5 * v0 / (2 * r)  # 角速度 (rad/s)

    theta = np.radians(theta_degree)
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
    x, y = getTraj(10.5, 90-38)
    # 绘制轨迹
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='With Magnus Effect')
    plt.title('Projectile Trajectory with Air Drag and Magnus Effect')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Height (m)')
    plt.grid(True)
    plt.legend()
    plt.show()