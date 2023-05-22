"""
"""
import numpy as np
import pandas as pd


class MyLineReg:
    """ Linear regression
    """
    def __init__(
        self,
        n_iter: int = 100,
        learning_rate: float = 0.1,
        weights: np.array = None
    ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.weights = weights

    def __str__(self) -> str:
        return f'MyLineReg class: n_iter={self.n_iter}, learning_rate={self.learning_rate}'

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        verbose: int = False
    ) -> None:
        X.insert(0, 'w0', 1)
        self.weights = np.ones((1, X.columns.__len__()))
        X = X.to_numpy()
        y = y.to_numpy()
        np.dot(X, y)

    def get_coef(
        self
    ) -> np.array:
        return self.weights[1:]