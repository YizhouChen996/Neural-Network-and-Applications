import numpy as np

class SLP:
    def __init__(self, input_size, learning_rate):
        self.size = input_size
        self.lr = learning_rate
        self.w = np.random.randn((self.size)).reshape((self.size, 1))
        self.x = None
        self.out = None
        self.dw = None
    
    def forward(self, x):
        out = np.sign(self.w.T.dot(x))
        self.x = x
        self.out = out
        return out
    
    def backward(self, t):
        self.dw = self.lr*((t-self.out)*self.x)
        # print(self.x.shape)
        # print(self.dw.shape)
        dw = self.dw.copy().reshape((self.size, 1))
        return dw

    def train(self, x, t, iters):
        for i in range(iters):
            for j in range(x.shape[1]):
                y = self.forward(x[:, j])
                dw = self.backward(t[j].reshape((1,)))
                # print('w:{}'.format(self.w))
                # print('dw:{}'.format(dw))
                self.w += dw
        result = self.forward(x)
        return result



if __name__ == "__main__":
    a = np.random.randn((1))
    b = np.ones((2,1))
    c = a*b
    d = np.ones((1,2))
    print(d[0,1])
    print(c)
