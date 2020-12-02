import sys, os
sys.path.append(os.pardir)
from dbmoon import *
from slp import *

if __name__ == "__main__":
    dataset = DBMOON(1000, 10, 6, -2)
    dataA, dataB = dataset.gen_dbdata()
    data = np.hstack([dataA, dataB])
    # print(data.shape)
    train_data = data[:2, :]
    train_label = data[2, :]
    # print(train_data)

    slp = SLP(input_size=2, learning_rate=0.01)
    predict_result = slp.train(train_data, train_label, 10)
    # print(predict_result)

    res_old = 0
    test_x = []
    test_y = []
    for x in np.arange(-15, 25, 0.1):
        for y in np.arange(-15, 15, 0.1):
            r = slp.forward(np.array([[x],[y]]))
            if(res_old<=0 and r>=0):
                test_x.append(x)
                test_y.append(y)
            res_old = r

    fig1 = plt.figure()
    plt.plot(test_x, test_y, 'g--')
    plt.scatter(dataA[0, :], dataA[1, :], marker='x')
    plt.scatter(dataB[0, :], dataB[1, :], marker='+')
    plt.savefig('slp')
    
