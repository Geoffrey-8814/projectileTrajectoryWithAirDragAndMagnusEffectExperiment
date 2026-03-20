import numpy as np
import matplotlib.pyplot as plt

from ballistics import solve_projectile_from_polar
from project_params import load_project_params


# calculate initial angle
# import math
# print (math.atan(1.1206635236740112 -0.8241485953330994)/(2.72072696685791-2.9562156200408936))

PROJECT_NAME = "fuel_2026"
PROJECT = load_project_params(PROJECT_NAME)


def getTraj(v0, theta_degree=55):
    theta = np.radians(theta_degree)
    sol = solve_projectile_from_polar(
        v0=v0,
        theta=theta,
        params=PROJECT,
        t_span=(0, 10),
        hit_ground=True,
        max_step=0.01,
    )

    # 提取结果
    x = sol.y[0]
    y = sol.y[1]
    return x, y 
if __name__ == "__main__":
    x, y = getTraj(7.8788, 90-(180*(1/3.14)))
    # 绘制轨迹
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label='With Magnus Effect')
    plt.title('Projectile Trajectory with Air Drag and Magnus Effect')
    plt.xlabel('Horizontal Distance (m)')
    plt.ylabel('Vertical Height (m)')
    plt.grid(True)
    plt.legend()
    plt.show()