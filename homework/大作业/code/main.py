import sys, os
import numpy as np
import matplotlib.pyplot as plt
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
    
    # 生成测试数据集

    # 定义网络结构
    network = ThreeLayerNet(input_size=2, hidden_size1=10, hidden_size2=5, output_size=1)

    # 设置超参数
    iters_num = 300000
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
        
        loss = network.loss(x_train, y_label)
        # print(loss)
        mean_loss = np.sum(loss, axis=1) / loss.shape[1]
        print("The {} iteration'loss:{}".format(i, mean_loss))
        train_loss_list.append(mean_loss)
    
    plt.plot(train_loss_list)
    plt.title("Training Loss")
    plt.show()
    plt.savefig('loss')