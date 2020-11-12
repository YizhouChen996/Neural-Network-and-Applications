import numpy as np
import matplotlib.pyplot as plt

class singleNN:
    
    def __init__(self, input_size, lr, iterations, train_data, train_label):
        self.size = input_size
        self.lr = lr
        self.iterations = iterations
        
        self.x_train = train_data
        self.y_label = train_label
        self.y_train = None
        
        self.W = np.random.randn(self.size, 1)
        self.B = np.random.randn(1)

        self.loss_list = []
    
    def loss(self):
        self.y_train = self.x_train.dot(self.W) + self.B
        temp = (self.y_train-self.y_label)**2 / 2
        loss = np.sum(temp, axis=0)/temp.shape[0]      
        return loss  

    def train(self):
        for i in range(self.iterations):
            self.loss_list.append(self.loss())
            self.y_train = self.x_train.dot(self.W) + self.B
            dy = self.y_train - self.y_label
            dW = self.x_train.T.dot(dy)
            dB = np.sum(dy, axis=0)
            self.W -= self.lr * dW
            self.B -= self.lr * dB
        return self.W, self.B, self.loss_list
    
    def predict(self, x_test):
        y_test = x_test.dot(self.W) + self.B
        return y_test

if __name__ == "__main__":
   
    data = np.array([55.22, 56.34, 55.52, 55.53, 56.94, 
                    58.88, 58.18, 57.09, 58.38, 58.54,
                    57.72, 58.02, 57.81, 58.71, 60.84,
                    61.08, 61.74, 62.16, 60.80, 60.87,
                    0, 0, 0, 0, 0]) 
    A = np.zeros((13, 8))
    for i in range(A.shape[0]):
        A[i, 0:A.shape[1]] = data[i: i+A.shape[1]]
    x_data = A[:, : -1]
    y_label = A[:, -1].reshape(13, 1)

    network = singleNN(input_size=7, lr=0.000001, iterations=100000, train_data=x_data, train_label=y_label)
    W, B, LOSS = network.train()
    for i in range(5):
        input_data = data[13+i:20+i]
        data[20+i] = network.predict(input_data)
        print("第{}天的预测数据为：{}".format(21+i, data[20+i]))

    print("W is:{}".format(W))
    print("B is:{}".format(B))
    plt.plot(LOSS)
    plt.axis([0, len(LOSS), 0, 1])
    plt.show()