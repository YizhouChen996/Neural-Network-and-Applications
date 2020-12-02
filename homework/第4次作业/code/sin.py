import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from neuralnet import *

if __name__ == "__main__":
    # 生成训练数据集
    x_train = np.arange(-5, 5, 0.1).reshape((1, 100))
    y_train = np.sin(x_train)

    print(y_train.shape)

    # 定义网络结构
    network = OneLayerNet(input_size=1, hidden_size=5, output_size=1)

    # 设置超参数
    iters_num = 400000
    train_size = x_train.shape[1]
    batch_size = 2
    learning_rate = 0.0001
    train_loss_list = []
    iter_per_epoch = max(train_size / batch_size, 1)

    # 模型训练
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[:, batch_mask]
        y_batch = y_train[:, batch_mask]

        grad = network.gradient(x_batch, y_batch)

        for key in (network.params.keys()):
            network.params[key] -= learning_rate*grad[key]

        if i % iter_per_epoch == 0:
            loss = network.loss(x_train, y_train)
            # print(loss)
            mean_loss = np.sum(loss, axis=1) / loss.shape[1]
            print("No.{} epoches' loss:{}".format((int)(i/iter_per_epoch)+1, mean_loss))
            train_loss_list.append(mean_loss)
    
    params, t, y, z = network.get_weights(2)
    print('params:{}'.format(params))
    print('t:{}'.format(t))
    print('y:{}'.format(y))
    print('z:{}'.format(z))

    # 绘制损失函数图像
    fig2 = plt.figure()
    plt.plot(train_loss_list)
    plt.title("Training Loss")
    plt.savefig("sin_loss")
    plt.show()

    # 绘制拟合图像
    fig3 = plt.figure()
    test = network.predict(x_train)
    print(test)
    plt.title('sin(x)')
    plt.plot(np.arange(-5, 5, 0.1), test.reshape((100,)))
    plt.savefig('sin')

    '''
    # 绘制拟合后预测结果
    fig3 = plt.figure()
    ax2 = plt.axes(projection='3d')
    output2 = network.predict(input)
    OUT2 = output2.reshape((100, 100))

    ax2.plot_surface(X,Y,OUT2)
    plt.savefig('3d2')
    '''
    