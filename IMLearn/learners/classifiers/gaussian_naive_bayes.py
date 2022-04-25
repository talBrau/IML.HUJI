from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None
        self.m_, self.d_, self.k_ = 0, 0, 0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.m_ = X.shape[0]
        self.d_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.k_ = self.classes_.shape[0]

        self.mu_ = np.zeros((self.k_, self.d_))
        self.vars_ = np.zeros((self.k_, self.d_))
        self.pi_ = np.zeros(self.k_)

        for i, k in enumerate(self.classes_):
            x_k = X[y == k]
            self.mu_[i] = np.mean(x_k)
            self.pi_[i] = x_k.shape[0] / self.m_
            self.vars_[i] = np.var(x_k, axis=0, ddof=1)

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
        pred = np.zeros((self.k_, self.m_))
        for i in range(self.classes_.shape[0]):
            log_pi = np.log(self.pi_[i])
            s = np.sum(((X - self.mu_[i]) ** 2) / self.vars_[i] + np.log(self.vars_[i]), axis=1)
            pred[i] = log_pi - 0.5 * s
        k_max = np.argmax(pred, axis=0)
        return self.classes_[k_max]

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        liklyhoods = np.zeros((self.k_, self.m_))
        for i in range(self.k_):
            x_mu = X - self.mu_[i]
            liklyhoods[i] = self.pi_[i] * np.product(
                np.exp(-x_mu ** 2 / 2 * self.vars_[i]) / np.square(2 * np.pi) * self.vars_[i])
        return liklyhoods.T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(self._predict(X), y)
