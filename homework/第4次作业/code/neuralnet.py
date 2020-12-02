import numpy as np
from layers import *
from collections import OrderedDict


class ThreeLayerNet:
    
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std = 0.1):
        # 初始化权重
        self.params = {}
        
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size1)
        self.params['B1'] = np.zeros((hidden_size1, 1))
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size1, hidden_size2)
        self.params['B2'] = np.zeros((hidden_size2, 1)) 
        self.params['W3'] = weight_init_std*np.random.randn(hidden_size2, output_size)
        self.params['B3'] = np.zeros((output_size, 1))

        # 初始化隐藏层
        self.layers = OrderedDict()
        
        self.layers['fc1'] = FC(self.params['W1'], self.params['B1'])
        self.layers['relu1'] = Relu()
        self.layers['fc2'] = FC(self.params['W2'], self.params['B2'])
        self.layers['relu2'] = Relu()
        self.layers['fc3'] = FC(self.params['W3'], self.params['B3'])

        self.lastlayer = Loss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)
    
    def gradient(self, x, t):
        # 前向传播
        self.loss(x, t)

        # 反向传播
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 保存偏导
        grads = {}
        grads['W1'], grads['B1'] = self.layers['fc1'].dW, self.layers['fc1'].dB
        grads['W2'], grads['B2'] = self.layers['fc2'].dW, self.layers['fc2'].dB
        grads['W3'], grads['B3'] = self.layers['fc3'].dW, self.layers['fc3'].dB

        return grads

class OneLayerNet:
    
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.001):
        # 初始化权重
        self.params = {}
        
        self.params['W1'] = weight_init_std*np.random.randn(input_size, hidden_size)
        self.params['B1'] = np.zeros((hidden_size, 1))
        self.params['W2'] = weight_init_std*np.random.randn(hidden_size, output_size)
        self.params['B2'] = np.zeros((output_size, 1)) 

        # 初始化隐藏层
        self.layers = OrderedDict()
        
        self.layers['fc1'] = FC(self.params['W1'], self.params['B1'])
        self.layers['relu1'] = Relu()
        self.layers['fc2'] = FC(self.params['W2'], self.params['B2'])

        self.lastlayer = Loss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x
    
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastlayer.forward(y, t)
    
    def gradient(self, x, t):
        # 前向传播
        self.loss(x, t)

        # 反向传播
        dout = 1
        dout = self.lastlayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 保存偏导
        grads = {}
        grads['W1'], grads['B1'] = self.layers['fc1'].dW, self.layers['fc1'].dB
        grads['W2'], grads['B2'] = self.layers['fc2'].dW, self.layers['fc2'].dB
        return grads
    
    def get_weights(self, x):
        t = self.layers['fc1'].forward(x)
        y = self.layers['relu1'].forward(t)
        z = self.layers['fc2'].forward(y)
        params = self.params
        return params, t, y, z


if __name__ == "__main__":
    pass
