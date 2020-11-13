import numpy as np

if __name__ == "__main__":
    x = np.array([[-1,-2],[3,4]])
    y = np.array(x > 0, dtype=np.int)
    print(y)