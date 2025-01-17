from __future__ import annotations
from typing import NoReturn
from IMLearn.base import BaseEstimator
import numpy as np
from numpy.linalg import pinv
from IMLearn.metrics import mean_square_error


class LinearRegression(BaseEstimator):
    """
    Linear Regression Estimator

    Solving Ordinary Least Squares optimization problem
    """

    def __init__(self, include_intercept: bool = True) -> LinearRegression:
        """
        Instantiate a linear regression estimator

        Parameters
        ----------
        include_intercept: bool, default=True
            Should fitted model include an intercept or not

        Attributes
        ----------
        include_intercept_: bool
            Should fitted model include an intercept or not

        coefs_: ndarray of shape (n_features,) or (n_features+1,)
            Coefficients vector fitted by linear regression. To be set in
            `LinearRegression.fit` function.
        """
        super().__init__()
        self.include_intercept_, self.coefs_ = include_intercept, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit Least Squares model to given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----
        Fits model with or without an intercept depending on value of `self.include_intercept_`
        """
        if self.include_intercept_:
            ones = np.ones((len(X), 1), dtype=int)
            X = np.concatenate((ones, X), axis=1)  # add a ones colomn to the left of X
        self.coefs_ = np.linalg.pinv(X) @ y

        # self.coefs_ =np.transpose(np.linalg.pinv(X.T)) @ y




    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        if self.include_intercept_:
            ones = np.ones((len(X), 1), dtype=int)
            X = np.concatenate((ones, X), axis=1)  # add a ones colomn to the left
        return X@self.coefs_

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under MSE loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under MSE loss function
        """
        if self.include_intercept_:
            ones = np.ones((len(X), 1), dtype=int)
            X = np.concatenate((ones, X), axis=1)  # add a ones column to the left of X
        return mean_square_error(y,X@self.coefs_)
