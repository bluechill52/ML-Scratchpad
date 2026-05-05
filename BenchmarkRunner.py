from LogisticRegression import LogisticRegression
from typing import Dict, List
from time import perf_counter
import numpy as np
from EvalHelpers import compute_all_metrics, \
                        get_best_threshold


class BenchmarkRunner:
    def __init__(self, benchmarkData: Dict,
                 warmupIters: int = 100,
                 benchmarkIters: int = 500):
        self._inputs = benchmarkData['valX']
        self._labels = benchmarkData['valY']
        self._warmupIters = warmupIters
        self._benchmarkIters = benchmarkIters
        self._results = []
    
    def benchmarkSingleModel(self, model: LogisticRegression):
        # Warm up the cache
        for _ in range(self._warmupIters):
            _ = model.predict_proba(self._inputs)
        
        latencies = []
        # Run the benchmark on the model
        for _ in range(self._benchmarkIters):
            startTime = perf_counter()
            probScores = model.predict_proba(self._inputs)
            deltaTime = (perf_counter() - startTime) * 1000
            latencies.append(deltaTime)
        
        latenciesNp = np.array(latencies)

        p99 = np.percentile(latenciesNp, 99)
        median = np.median(latenciesNp)
        mean = np.mean(latenciesNp)

        best_threshold = get_best_threshold(probScores=probScores, trueLabels=self._labels, metric='f1')
        predLabels = (probScores >= best_threshold).astype(int)
        metrics = compute_all_metrics(predLabels, self._labels)

        return {
            'accuracy' : metrics['accuracy'],
            'precision' : metrics['precision'],
            'recall' : metrics['recall'],
            'f1' : metrics['f1'],
            'best_threshold' : best_threshold,
            'latency_p99_ms' : p99,
            'latency_median_ms' : median,
            'latency_mean_ms' : mean,
        }

    
    def run(self, models: Dict[str, LogisticRegression]):
        for name, model in models.items():
            result = self.benchmarkSingleModel(model)
            result['model_name'] = name

            print(result)
            self._results.append(result)
        
    