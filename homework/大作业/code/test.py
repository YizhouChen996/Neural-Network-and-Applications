import numpy as np
import math

a = np.ones((5, 3))
b = np.sum(a, axis=1, keepdims=True)
print(a)
print(b.shape)

"""
x = np.arange(-5, 5, 0.1)
y = np.arange(-5, 5, 0.1)
data = np.zeros((2, x.size*y.size))
for i in range(x.size):
    for j in range(y.size):
        data[0, 100*i+j] = x[i]
        data[1, 100*i+j] = y[j]
        # print(100*i+j)
        # print(data[:, 100*i+j])
label = np.zeros(x.size*y.size)
for i in range(label.size):
    label[i] = math.sin(data[0, i]) - math.cos(data[1, i])
label.reshape((1, 10000))
print(label)
# label = math.sin(x) - math.cos(y)
"""
"""
v = np.maximum(0, x)
print(v)
s = np.zeros(x.shape)
s[x>=0] = 1
print(s)
"""