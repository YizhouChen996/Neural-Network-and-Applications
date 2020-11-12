import numpy as np
if __name__ == "__main__":
    a = np.array([1, 2])
    b = np.array([4, 5])
    temp = (a-b)**2/2
    loss = np.sum(temp, axis=0)/temp.shape[0]
    print(loss)