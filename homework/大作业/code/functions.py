import numpy as np

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def sigmoid_grad(x):
    return (1.0-sigmoid(x))*sigmoid(x)

def relu(x):
    return np.maximum(0, x)

def relu_grad(x):
    grad = np.zeros(x.shape)
    grad[x>=0] = 1
    return grad



if __name__ == "__main__":
    pass