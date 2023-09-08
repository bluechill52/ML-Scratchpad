import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def run():
    num_samples = 1000
    num_dims = 2
    
    X = np.random.uniform(0, 50, [num_samples, num_dims])
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    
    w1_true = np.random.uniform(1, 50)
    w2_true = np.random.uniform(1, 50)
    b_true = np.random.uniform(1, 50)
    
    Y = w1_true * X[:, 0] + w2_true * X[:, 1] + b_true
    
    trainX, testX, trainY, testY = train_test_split(X, Y, train_size=0.8)
    
    epochs = 3000
    alpha = 0.001
    
    w1 = 0.0
    w2 = 0.0
    b = 0.0
    for epoch in range(epochs):
        mse = 0.0
        de_dw1 = 0.0
        de_dw2 = 0.0
        de_db = 0.0
        for i in range(trainX.shape[0]):
            y_true = trainY[i]
            x1 = trainX[i][0]
            x2 = trainX[i][1]
            y_pred = w1 * x1 + w2 * x2 + b
            
            mse += (y_true - y_pred) ** 2
            de_dw1 += -2.0 * (y_true - y_pred) * x1
            de_dw2 += -2.0 * (y_true - y_pred) * x2
            de_db += -2.0 * (y_true - y_pred)
                
        mse /= trainX.shape[0]
        de_dw1 /= trainX.shape[0]
        de_dw2 /= trainX.shape[0]
        de_db /= trainX.shape[0]
            
        w1 -= alpha * de_dw1
        w2 -= alpha * de_dw2
        b -= alpha * de_db
        
        print(f'Epoch {epoch} mse {mse} {w1_true - w1} {w2_true - w2} {b_true - b}')
        
        
    print('ddd')
    
    
if __name__ == '__main__':
    run()