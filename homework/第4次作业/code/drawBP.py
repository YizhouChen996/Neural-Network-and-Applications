import sys, os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math
from neuralnet import *
from dbmoon import *

if __name__ == "__main__":

    dataset = DBMOON(1000, 10, 6, -2)
    dataA, dataB = dataset.gen_dbdata()
    data = np.hstack([dataA, dataB])
    # print(data.shape)
    x_train = data[:2, :]
    y_label = data[2, :]
    # print(train_data)

    # 定义网络结构
    network = ThreeLayerNet(input_size=2, hidden_size1=10, hidden_size2=5, output_size=1)

    # 设置超参数
    iters_num = 500000
    train_size = x_train.shape[1]
    batch_size = 100
    learning_rate = 0.00001
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
   
    res_old = 0
    test_x = []
    test_y = []
    for x in np.arange(-15, 25, 0.1):
        for y in np.arange(-15, 15, 0.1):
            r = network.predict(np.array([[x],[y]]))
            if(res_old<=0 and r>=0):
                test_x.append(x)
                test_y.append(y)
            res_old = r

    fig1 = plt.figure()
    plt.plot(test_x, test_y, 'g--')
    plt.scatter(dataA[0, :], dataA[1, :], marker='x')
    plt.scatter(dataB[0, :], dataB[1, :], marker='+')
    plt.savefig('bp')