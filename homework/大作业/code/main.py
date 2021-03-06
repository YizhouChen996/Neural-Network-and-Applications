import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from neuralnet import *

if __name__ == "__main__":
    # 生成训练数据集
    data1 = np.arange(-5, 5, 0.1)
    data2 = np.arange(-5, 5, 0.1)
    x_train = np.zeros((2, data1.size*data2.size))
    for i in range(data1.size):
        for j in range(data2.size):
            x_train[0, 100*i+j] = data1[i]
            x_train[1, 100*i+j] = data2[j]
    y_label = np.zeros(data1.size*data2.size)
    for i in range(y_label.size):
        y_label[i] = math.sin(x_train[0, i]) - math.cos(x_train[1, i])
    y_label.reshape((1, data1.size*data2.size))

    # 定义网络结构
    network = ThreeLayerNet(input_size=2, hidden_size1=10, hidden_size2=5, output_size=1)
    
    '''
    # 画出模型初始的预测效果
    fig1 = plt.figure()
    ax1 = plt.axes(projection='3d')
    X, Y = np.meshgrid(data1, data2)
    x = X.reshape((1, -1))
    y = Y.reshape((1, -1))
    input = np.vstack((x, y))
    output = network.predict(input)
    OUT = output.reshape((100, 100))

    ax1.plot_surface(X,Y,OUT)
    plt.savefig('3d1')
    '''

    # 设置超参数
    iters_num = 5000000
    train_size = x_train.shape[1]
    batch_size = 100
    learning_rate = 0.0001
    train_loss_list = []
    iter_per_epoch = max(train_size / batch_size, 1)

    # 模型训练
    for i in range(iters_num):
        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[:, batch_mask]
        y_batch = y_label[batch_mask]

        grad = network.gradient(x_batch, y_batch)

        for key in (network.params.keys()):
            network.params[key] -= learning_rate*grad[key]

        if i % iter_per_epoch == 0:
            loss = network.loss(x_train, y_label)
            # print(loss)
            mean_loss = np.sum(loss, axis=1) / loss.shape[1]
            print("No.{} epoches' loss:{}".format((int)(i/iter_per_epoch)+1, mean_loss))
            train_loss_list.append(mean_loss)
   
    # 绘制损失函数图像
    fig2 = plt.figure()
    plt.plot(train_loss_list)
    plt.title("Training Loss")
    plt.show()
    plt.savefig('tanh_5')
    
    '''
    # 绘制拟合后预测结果
    fig3 = plt.figure()
    ax2 = plt.axes(projection='3d')
    output2 = network.predict(input)
    OUT2 = output2.reshape((100, 100))

    ax2.plot_surface(X,Y,OUT2)
    plt.savefig('3d2')
    '''