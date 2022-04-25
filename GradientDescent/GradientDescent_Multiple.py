import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


# 求fx函数值

def fx(x, y):
    return (x - 10)**2 + (y - 10)**2


def gradient_descent():
    # 迭代次数
    times = 100

    # 学习率
    alpha = 0.05

    # x, y 初始值
    x = 20
    y = 20

    # 将画布设置为3D
    fig = Axes3D(plt.figure())

    # 设置X轴取值范围
    axis_x = np.linspace(0, 20, 100)

    # 设置Y轴取值范围
    axis_y = np.linspace(0, 20, 100)

    # 将数据转化为网格数据
    axis_x, axis_y = np.meshgrid(axis_x, axis_y)

    # 计算z轴值
    z = fx(axis_x, axis_y)

    fig.set_xlabel("X", fontsize=14)
    fig.set_ylabel("Y", fontsize=14)
    fig.set_zlabel("Z", fontsize=14)

    # 设置3D图的俯视角度， 方便查看梯度下降曲线
    fig.view_init(elev=60, azim=300)
    # 作出底图
    fig.plot_surface(axis_x, axis_y, z, rstride=1,
                     cstride=1, cmap=plt.get_cmap('rainbow'))

    for i in range(times):
        x1 = x
        y1 = y

        f1 = fx(x, y)

        print('第%d次迭代： x=%f, y=%f, f(x, y)=%f')

        x = x - alpha * 2 * (x - 10)
        y = y - alpha * 2 * (y - 10)

        f = fx(x, y)
        fig.plot([x1, x], [y1, y], [f1, f], 'ko', lw=2, ls='-')

    plt.show()


if __name__ == '__main__':
    gradient_descent()
