from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Dict
from LogisticRegression import LogisticRegression
from BenchmarkRunner import BenchmarkRunner

if __name__ == '__main__':
    # Create the dataset
    X, Y = make_classification(n_samples=1000,
                               n_features=10,
                               random_state=42)
    
    trainX, valX, trainY, valY = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Compute mean and std from training data for normalization
    mean, std = np.mean(trainX, axis=0), np.std(trainX, axis=0)

    # Normalize training and validation data to 0 mean and unit std
    trainX = (trainX - mean) / (std + 1e-9)
    valX = (valX - mean) / (std + 1e-9)


    models = {
        'modelA' : LogisticRegression(0.1, 10, 0.01),
        'modelB' : LogisticRegression(0.2, 100, 0.02),
        'modelC' : LogisticRegression(0.3, 500, 0.03),
    }

    # Train the models
    for model in models.values():
        model.fit(trainX, trainY)
    
    # Benchmark the models
    benchmarkData = {
        'valX' : valX,
        'valY' : valY,
    }

    benchmark = BenchmarkRunner(benchmarkData)
    benchmark.run(models)

    # Verify shape of training and validation data
    # print(trainX.shape, valX.shape, trainY.shape, valY.shape)

    # model = LogisticRegression(lr=0.1, epochs=500, l2=0.01)
    # model.fit(trainX, trainY)







