import numpy as np
import random

class neuron:
    def __init__(self):
        self.w0 = (random.random()- 0.5)*10
        self.w1 = (random.random()- 0.5)*10
        self.w2 = (random.random()- 0.5)*10
        self.w3 = (random.random()- 0.5)*10
        self.w4 = (random.random()- 0.5)*10
        self.b = 50 * (random.random() - 0.5)

    def output(self, i0, i1, i2, i3, i4):
        return self.w0 * i0 + self.w1 * i1 + self.w2 * i2 + self.w3 * i3 + self.w4 * i4 + self.b

    def train(self, n, data):
        delta_w0 = 0
        delta_w1 = 0
        delta_w2 = 0
        delta_w3 = 0
        delta_w4 = 0
        delta_b = 0
        for i in range(data.shape[0]):
            delta_w0 += n * (self.output(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4]) - data[i][5])*data[i][0]
            delta_w1 += n * (self.output(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4]) - data[i][5])*data[i][1]
            delta_w2 += n * (self.output(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4]) - data[i][5])*data[i][2]
            delta_w3 += n * (self.output(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4]) - data[i][5])*data[i][3]
            delta_w4 += n * (self.output(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4]) - data[i][5])*data[i][4]
            delta_b  += n * (self.output(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4]) - data[i][5])
        self.w0 = self.w0 - delta_w0
        self.w1 = self.w1 - delta_w1
        self.w2 = self.w2 - delta_w2
        self.w3 = self.w3 - delta_w3
        self.w4 = self.w4 - delta_w4
        self.b = self.b - delta_b
        # print(self.w0, " ", self.w1, " ",self.w2, " ",self.w3, " ",self.w4," ",self.b)
    
    def loss(self,data):
        loss_value = 0
        for i in range(data.shape[0]):
            loss_value += (self.output(data[i][0],data[i][1],data[i][2],data[i][3],data[i][4]) - data[i][5]) ** 2
        return loss_value
    
if __name__ == '__main__':
    share = [55.22,56.34,55.52,55.53,56.94,58.88,58.18,57.09,58.38,38.54,57.72,58.02,57.81,58.71,60.84,61.08,61.74,62.16,60.80,58.54]
    data = np.zeros(shape = [15,6],dtype = float)
    neu = neuron()
    for i in range(15):
        for j in range(6):
            data[i][j] = share[i+j]

    for i in range(20000):
        n = 0.000001
        neu.train(n,data)
        # print(neu.loss(data))
    
    print(neu.w0)
    print(neu.w1)
    print(neu.w2)
    print(neu.w3)
    print(neu.w4)
    print(neu.b)
    
    for i in range(5):
        now = neu.output(share[i+10],share[i+11],share[i+12],share[i+13],share[i+14])
        print("day:" ,i+21," ",now)
        share.append(now)
