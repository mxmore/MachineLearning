# 根据给定样本求解出最佳θ组合

from distutils.log import error
from xml.etree.ElementInclude import XINCLUDE
import numpy as np
import matplotlib.pyplot as plt


# 预测目标变量y的值

def predict(theta_0, theta_1, x):
    y_predicted = theta_0 + theta_1 * x
    return y_predicted

# 遍历样本数据，计算偏差，使用批量梯度下降


def loop(m, theta_0, theta_1, x, y):
    sum1 = 0
    sum2 = 0
    error = 0

    for i in range(m):
        a = predict(theta_0, theta_1, x[i]) - y[i]
        b = (predict(theta_0, theta_1, x[i]) - y[i]) * x[i]
        error1 = a * a

        sum1 = sum1 + a
        sum2 = sum2 + b

        error = error + error1

    return sum1, sum2, error

# 批量梯度下降进行更新theta值


def batch_gradient_descent(x, y, theta_0, theta_1, alpha, m):
    gradient_1 = (loop(m, theta_0, theta_1, x, y)[1]/m)
    # 设定一个阈值， 当梯度的绝对值小于0.001时即不再更新
    while abs(gradient_1) > 0.001:
        sum1, sum2, error1 = loop(m, theta_0, theta_1, x, y)
        gradient_0 = (sum1/m)
        gradient_1 = (sum2/m)
        error = (error1/m)
        theta_0 = theta_0 - alpha*gradient_0
        theta_1 = theta_1 - alpha*gradient_1

    return (theta_0, theta_1, error)


if __name__ == '__main__':
    # 样本数据
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    y = [3, 4, 5, 5, 2, 4, 7, 8, 11, 8, 10, 11, 13, 13, 16, 17, 16, 17, 18, 20]

    # 样本数量
    m = 20

    # 学习率
    alpha = 0.01

    # 初始化theta_0, theta_1,
    theta_0 = 1
    theta_1 = 1
    error = 0

    theta_0 = batch_gradient_descent(x, y, theta_0, theta_1, alpha, m)[0]
    theta_1 = batch_gradient_descent(x, y, theta_0, theta_1, alpha, m)[1]
    error = batch_gradient_descent(x, y, theta_0, theta_1, alpha, m)[2]
    # theta_0, theta_1, error = batch_gradient_descent(
    #     x, y, theta_0, theta_1, alpha, m)

    print("The theta_0 is %f, the theta_1 is %f, the mean squared error is %f " % (
        theta_0, theta_1, error))

    # 新建一个画布
    plt.figure(figsize=(6, 4))
    # 绘制样本散点图
    plt.scatter(x, y, label="Y")
    # X轴范围
    plt.xlim(0, 21)
    # Y轴范围
    plt.ylim(0, 22)
    # X标签
    plt.xlabel("x", fontsize=14)
    # Y标签
    plt.ylabel("y", fontsize=14)

    x = np.array(x)
    y_predicted = np.array(theta_0 + theta_1*x)
    # 绘制拟合的函数图
    plt.plot(x, y_predicted, color='red')
    plt.show()
