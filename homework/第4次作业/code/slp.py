import numpy as np

class SLP:
    def __init__(self, input_size, learning_rate):
        self.size = input_size
        self.lr = learning_rate
        self.w = np.random.randn((self.size, 1))
        self.x = None
        self.out = None
        self.dw = None
    
    def forward(self, x):
        out = x.dot(self.w)
        self.x = x
        self.out = out
        return out
    
    def backward(self, t):
        self.dw = 



if __name__ == "__main__":
    a = np.random.randn((2))
    b = np.ones((2,1))
    c = a.dot(b)
    print(a)
    print(b)
    print(c)
