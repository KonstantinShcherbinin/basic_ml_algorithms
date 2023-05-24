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
        if not {'w0'}.issubset(X.columns):
            X.insert(0, 'w0', 1)
        self.weights = np.ones((X.columns.__len__(), 1))
        X = X.to_numpy()
        y = np.resize(y, (len(y), 1))
        iteration = 1
        while iteration <= self.n_iter:
            predicted = np.array(np.dot(X, self.weights))
            gradient = (2/len(predicted)) * np.dot(X.T, (predicted - y))
            self.weights = self.weights - self.learning_rate * gradient
            if verbose:
                if iteration == 1:
                    print(f'start | loss: {gradient}')
                if iteration % verbose == 0:
                    print(f'{iteration} | loss: {gradient}')
            iteration += 1

    def get_coef(
        self
    ) -> np.array:
        return self.weights[1:]