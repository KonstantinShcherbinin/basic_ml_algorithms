import numpy as np

def metric_calc(
        y: np.array,
        pred: np.array,
        metric: str
) -> float:
def metric_calc(y, pred, metric):
        if metric == 'mae':
            return sum(abs(y - pred)) / len(y)
        elif metric == 'mse':
            return sum(np.square(y - pred)) / len(y)
        elif metric == 'rmse':
            return (sum(np.square(y - pred)) / len(y)) ** 0.5
        elif metric == 'r2':
            return 1 - (sum(np.square(y - pred)) / sum(np.square(y - np.mean(y))))
        elif metric == 'mape':
            return 100 * sum(abs((y - pred) / y)) / len(y)