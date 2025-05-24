import pandas as pd
import matplotlib.pyplot as plt

import traj

speeds = [40, 50, 60]

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
for speed in speeds:


    x1, y1 = readCoordinates(speed, 1)
    x2, y2 = readCoordinates(speed, 2)
    x3, y3 = readCoordinates(speed, 3)


    # for i in range(3):
    #     dataset.append(pd.read_csv(f'dataset\\{speed}-{i+1}annotations.csv'))
        
    x, y = traj.getTraj(speed)

    # 绘制图形（修改后的部分）
    plt.figure(figsize=(12, 7))
    plt.plot(x, y, 'b-', linewidth=2, label='Simulation with Magnus')  # 模拟结果
    plt.plot(x1, y1, 'r--o', markersize=6, label='Test Data Group 1')  # 测试组1
    plt.plot(x2, y2, 'g--s', markersize=6, label='Test Data Group 2')  # 测试组2
    plt.plot(x3, y3, 'm--D', markersize=6, label='Test Data Group 3')  # 测试组3

    plt.title(f'Projectile Trajectory: Simulation vs Experimental Data (rotor speed = {speed})', fontsize=14)
    plt.xlabel('Horizontal Distance (m)', fontsize=12)
    plt.ylabel('Vertical Height (m)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)

    # 自动调整坐标轴范围
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # 显示图形
    plt.show()





