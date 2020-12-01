import numpy as np
import matplotlib.pyplot as plt

class DBMOON:
    def __init__(self, n, r, w, d):
        self.n = n
        self.r = r
        self.w = w
        self.d = d

        self.dataA = None
        self.dataB = None
        self.dataDB = None
    
    def gen_dbdata(self):
        theta1 = np.random.uniform(0, np.pi, size=self.n)
        theta2 = np.random.uniform(-np.pi, 0, size=self.n)
        w1 = np.random.uniform(-self.w/2, self.w/2, size=self.n)
        w2 = np.random.uniform(-self.w/2, self.w/2, size=self.n)
        one = np.ones_like(theta1)

        self.dataA = np.array([(self.r+w1)*np.cos(theta1), (self.r+w2)*np.sin(theta1), one])
        self.dataB = np.array([self.r+(self.r+w2)*np.cos(theta2), -self.d+(self.r+w2)*np.sin(theta2), -one])

        dataA = self.dataA.copy()
        dataB = self.dataB.copy()

        return dataA, dataB
    
    def plot(self):
        fig = plt.figure()
        plt.scatter(self.dataA[0, :], self.dataA[1, :], marker='x')
        plt.scatter(self.dataB[0, :], self.dataB[1, :], marker='+')
        plt.savefig("dbdata")

if __name__ == "__main__":
    dataset = DBMOON(1000, 10, 6, -2)
    dataA, dataB = dataset.gen_dbdata()
    print(dataB)
    dataset.plot()
