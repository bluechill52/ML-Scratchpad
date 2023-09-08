import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


LOOP = False

def run():
    num_samples = 1000
    num_dims = 100
    
    X = np.random.uniform(0, 50, [num_samples, num_dims])
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    w_true = np.random.uniform(1, 50, [num_dims, 1])
    b_true = np.random.uniform(1, 50)
    
    Y = np.dot(X, w_true) + b_true
    
    trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.8)
    
    epochs = 10000
    alpha = 0.001
    
    w = np.zeros([num_dims, 1])
    b = 0.0
    for epoch in range(epochs):
        mse = 0.0
        de_dw = 0.0
        de_db = 0.0
        
        y_true = trainY.reshape(-1, 1)
        y_pred = np.dot(trainX, w) + b
        mse = np.sum((y_true - y_pred) ** 2, axis=0) / trainX.shape[0]
        de_dw = np.sum(np.multiply(-2.0 * (y_true - y_pred), trainX), axis=0).reshape(-1, 1) / trainX.shape[0]
        de_db = np.sum(-2.0 * (y_true - y_pred), axis=0).reshape(-1, 1) / trainX.shape[0]
            
        w -= alpha * de_dw
        b -= alpha * de_db
        
        print(f'Epoch {epoch} mse {mse}')
        
    pltX = np.array(list(range(testX.shape[0])))
    pltY = np.dot(testX, w) + b
    plt.scatter(pltX, testY)
    plt.scatter(pltX, pltY)
    plt.show()
    
    print('ddd')
    
    
if __name__ == '__main__':
    run()