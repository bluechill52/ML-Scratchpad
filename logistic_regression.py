import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


if __name__ == '__main__':
    data = pd.read_csv('dmv_dataset.csv')
    results = data['Results'].to_numpy().reshape(-1, 1)
    scores = data[['DMV_Test_1', 'DMV_Test_2']].to_numpy()
    
    # Mean center and unit std - normalization
    scores = (scores - np.mean(scores, axis=0)) / np.std(scores, axis=0)
    
    print(data.head())
    
    epochs = 5
    alpha = 0.001
    w = np.zeros([scores.shape[1], 1])
    b = 0.0
    
    for epoch in range(epochs):
        y_true = results
        y_pred = sigmoid(np.dot(scores, w) + b)
        L = -np.sum(y_true * np.log(y_pred) + (1.0 - y_true) * np.log(1.0 - y_pred), axis=0) / scores.shape[0]
        dL_dw = np.dot(scores.T, (y_pred - y_true))
        dL_db = np.sum(y_pred - y_true, axis=0) / scores.shape[0]
        
        w -= alpha * dL_dw
        b -= alpha * dL_db
        
        print(f'Epoch {epoch} log loss {L}')
    
    # Plot the data
    positiveX = scores[:, 0][np.where(results == 1)[0]]
    positiveY = scores[:, 1][np.where(results == 1)[0]]
    
    negativeX = scores[:, 0][np.where(results == 0)[0]]
    negativeY = scores[:, 1][np.where(results == 0)[0]]
    
    plt.scatter(positiveX, positiveY)
    plt.scatter(negativeX, negativeY)
    
    # Plot the decision boundary
    # sigmoid(w * x + b) >= 0.5 -> class 1
    # sigmoid(w * x + b) >= 0.5 -> class 0
    # => w * x + b >= 0 -> class 1 else class 0
    # Decision line - w * x + b == 0
    # => w1 * x1 + w2 * x2 + b = 0
    # => x2 = (-b - w1 * x1) / w2
    
    minX = min(scores[:, 0])
    maxX = max(scores[:, 0])
    
    minY = (-b - w[0] * minX) / w[1]
    maxY = (-b - w[0] * maxX) / w[1]
    
    plt.plot([minX, maxX], [minY, maxY])
    plt.show()
        
    print('Done')