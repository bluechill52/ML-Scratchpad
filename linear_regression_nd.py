import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd



def predict(X, Y, w, b):
    num_samples = X.shape[0]
    mse = 0.0
    for i in range(num_samples):
        y_true = Y[i]
        x = X[i, :].reshape(-1, 1)
        y_pred = np.dot(w.T, x) + b
        
        mse += (y_true - y_pred) ** 2
    
    mse /= num_samples
    
    return mse

    
if __name__ == '__main__':
    data = pd.read_csv('california_housing_train.csv', index_col=False)

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)
    dataX = data.drop('median_house_value', axis=1).to_numpy()
    dataY = data['median_house_value'].to_numpy()

    num_rows = dataX.shape[0]
    split = [80, 10, 10]

    train_idx_lim = int(num_rows * split[0] / 100)
    val_idx_lim = int(num_rows * (split[0] + split[1]) / 100)

    trainX = dataX[:train_idx_lim, :]
    trainY = dataY[:train_idx_lim]

    valX = dataX[train_idx_lim:val_idx_lim, :]
    valY = dataY[train_idx_lim:val_idx_lim]

    testX = dataX[val_idx_lim:, :]
    testY = dataY[val_idx_lim:]

    # Normalize features
    trainX_normed = trainX / trainX.max(axis=0)
    trainY_normed = trainY / trainY.max(axis=0)

    valX_normed = valX / valX.max(axis=0)
    valY_normed = valY / valY.max(axis=0)

    testX_normed = testX / testX.max(axis=0)
    testY_normed = testY / testY.max(axis=0)

    epochs = 100
    alpha = 0.01

    num_dims = trainX_normed.shape[1]
    num_train_samples = trainX_normed.shape[0]

    w = np.zeros([num_dims, 1])
    b = 0.0

    for epoch in range(epochs):
        de_dw = np.zeros_like(w)
        de_db = 0.0
        mse = 0.0
        
        for i in range(num_train_samples):
            y_true = trainY_normed[i]
            x = trainX_normed[i, :].reshape(-1, 1)
            y_pred = np.dot(w.T, x) + b
            
            mse += (y_true - y_pred) ** 2
            de_dw += -2.0 * (y_true - y_pred) * x
            de_db += -2.0 * (y_true - y_pred)
        
        mse /= num_train_samples
        de_dw /= num_train_samples
        de_db /= num_train_samples
        
        w -= alpha * de_dw
        b -= alpha * de_db
        
        # Check error on validation set
        val_mse = predict(valX_normed, valY_normed, w, b)
        
        print(f'Epoch {epoch} mse {mse} val mse {val_mse}')

    print('Done')
