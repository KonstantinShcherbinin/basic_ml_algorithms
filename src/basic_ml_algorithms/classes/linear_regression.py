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
    regularization_list = ['l1', 'l2', 'elasticnet']

    def __init__(
            self,
            n_iter: int = 100,
            learning_rate: float = 0.1,
            weights: Optional[np.array] = None,
            metric: Optional[str] = None,
            err: Optional[float] = None,
            reg: Optional[str] = None,
            l1_coef: Optional[float] = 0,
            l2_coef: Optional[float] = 0
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights
        if metric and metric not in MyLineReg.metric_list:
            raise ValueError(f"{metric}-metric isn't in the list of available values")
        self.metric = metric
        self.err = err
        if reg and reg not in MyLineReg.regularization_list:
            raise ValueError(f"Regularization-type - {reg} isn't in the list of available values")
        self.reg = reg
        self.l1_coef = l1_coef
        self.l2_coef = l2_coef

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
            sgn = self.weights.copy()
            sgn = np.where(sgn < 0, -1, sgn)
            sgn = np.where(sgn > 0, 1, sgn)
            sgn = np.where(sgn == 0, 0, sgn)
            l1_weights = self.l1_coef * sgn
            l2_weights = self.l2_coef * self.weights * 2
            reg_weights = l1_weights if self.reg == 'l1' else \
                l2_weights if self.reg == 'l2' else \
                l1_weights + l2_weights if self.reg == 'elasticnet' else 0
            gradient = (2/len(predicted)) * np.dot(X.T, (predicted - y)) + reg_weights
            self.weights -= self.learning_rate * gradient
            #err = metric_calc(y, predicted, self.metric)
            self.err = MyLineReg.metric_calc(y, np.array(np.dot(X, self.weights)), self.metric)
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