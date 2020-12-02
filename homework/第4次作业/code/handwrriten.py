import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    '''
    digits_train = pd.read_csv("../dataset/optdigits.tra", header=None)
    digits_test = pd.read_csv("../dataset/optdigits.tes", header=None)
    # print(digits_train)
    x_train = np.array(digits_train[np.arange(64)])
    y_train = np.array(digits_train[64])
    x_test = np.array(digits_test[np.arange(64)])
    y_test = np.array(digits_test[64])
    # print(y_test.shape)

    clf = MLPClassifier(hidden_layer_sizes=(200,))
    clf.fit(x_train, y_train)
    # print(clf.predict(x_test))
    print(clf.score(x_test,y_test))
    '''
    x = np.arange(10, 110, 10)
    y = np.array([0.933 ,0.956, 0.954 , 0.958 , 0.956, 0.954, 0.963, 0.960, 0.962, 0.968 ])
    fig = plt.figure()
    plt.plot(x, y)
    plt.title("Accuracy Rate")
    plt.savefig('result')
