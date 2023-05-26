"""
"""
from typing import Optional, Any
import numpy as np
import pandas as pd

#from utils import metric_calc


class MyLineReg:
    """ Linear regression
    """
    metric_list = ['mae', 'mse', 'rmse', 'mape', 'r2']

    def __init__(
            self,
            n_iter: int = 100,
            learning_rate: float = 0.1,
            weights: Optional[np.array] = None,
            metric: Optional[str] = None,
            err: Optional[float] = None
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        if metric not in MyLineReg.metric_list:
            raise ValueError(f"{metric}-metric isn't in the list of available values")
        self.metric = metric
        self.err = err

    def __str__(self) -> str:
        return f'MyLineReg class: n_iter={self.n_iter}, \
            learning_rate={self.learning_rate}'

    def fit(
            self,
            X: pd.DataFrame,
            y: pd.Series,
            verbose: int = False
    ) -> None:
        if not {'w0'}.issubset(X.columns):
            X.insert(0, 'w0', 1)
        self.weights = np.ones((X.columns.__len__(), 1))
        X = X.to_numpy()
        y = np.resize(y, (len(y), 1))
        iteration = 1
        while iteration <= self.n_iter:
            predicted = np.array(np.dot(X, self.weights))
            #err = metric_calc(y, predicted, self.metric)
            self.err = MyLineReg.metric_calc(y, predicted, self.metric)
            gradient = (2/len(predicted)) * np.dot(X.T, (predicted - y))
            self.weights = self.weights - self.learning_rate * gradient
            if verbose:
                if self.metric:
                    if iteration == 1:
                        print(f'start | loss: {gradient} | {self.metric}: {self.err}')
                    if iteration % verbose == 0:
                        print(f'{iteration} | loss: {gradient} | {self.metric}: {self.err}')
                else:
                    if iteration == 1:
                        print(f'start | loss: {gradient}')
                    if iteration % verbose == 0:
                        print(f'{iteration} | loss: {gradient}')
            iteration += 1

    def get_coef(
            self
    ) -> np.array:
        return self.weights[1:]

    def predict(
            self,
            X: pd.DataFrame
    ) -> np.array:
        if not {'w0'}.issubset(X.columns):
            X.insert(0, 'w0', 1)
        X = X.to_numpy()
        return np.dot(X, self.weights)

    def metric_calc(
            y: np.array,
            pred: np.array,
            metric: str
    ) -> float:
        if metric == 'mae':
            return np.sum(np.fabs(y - pred)) / len(y)
        elif metric == 'mse':
            return np.sum(np.square(y - pred)) / len(y)
        elif metric == 'rmse':
            return (np.sum(np.square(y - pred)) / len(y)) ** 0.5
        elif metric == 'r2':
            return 1 - (np.sum(np.square(y - pred)) / np.sum(np.square(y - np.mean(y))))
        elif metric == 'mape':
            return 100 * np.sum(np.fabs((y - pred) / y)) / len(y)

    def get_best_score(self) -> float:
        return self.err