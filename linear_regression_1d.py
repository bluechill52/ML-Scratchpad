import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


num_samples = 100
X = np.linspace(-1, 1, num_samples)
Y = 7.3453 * X + 162.5456

w = 0.0
b = 0.0
epochs = 15000
alpha = 0.001

for epoch in range(epochs):
    de_dw = 0.0
    de_db = 0.0
    mse = 0.0
    
    for i in range(num_samples):
        x = X[i]
        y_true = Y[i]
        y_pred = w * x + b
        mse += (y_pred - y_true) ** 2
        de_dw += -2.0 * (y_true - y_pred) * x
        de_db += -2.0 * (y_true - y_pred)
        
    mse /= num_samples
    de_dw /= num_samples
    de_db /= num_samples
    
    w -= alpha * de_dw
    b -= alpha * de_db
    
    print(f'Epoch {epoch} -> error {mse} estimated {w} and {b}')
    
print('Done')