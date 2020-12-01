import numpy as np
from functions import *

class Relu:
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        self.mask = (x<=0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Tanh:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = tanh(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*(1-self.out**2)
        return dx
        
class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        out = sigmoid(x)
        self.out = out
        return out

    def backward(self, dout):
        dx = dout*(1.0-self.out)*self.out
        return dx

class FC:
    def __init__(self, W, B):
        self.W = W
        self.B = B

        self.x = None
        self.dW = None
        self.dB = None

    def forward(self, x):
        self.x = x
        out = self.W.T.dot(self.x)+self.B
        return out

    def backward(self, dout):
        dx = self.W.dot(dout)
        self.dW = self.x.dot(dout.T)
        self.dB = np.sum(dout, axis=1, keepdims=True)
        return dx

class Output:
    def __init__(self):
        self.y = None
        self.t = None
    
    def forward(self, y, t):
        self.y = y
        self.t = t
        loss = (self.y-self.t)**2
        return loss
    
    def backward(self, dout):
        dy = dout*2*(self.y - self.t)
        return dy
    
if __name__ == "__main__":
    pass