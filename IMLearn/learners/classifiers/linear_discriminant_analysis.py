from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None
        self.m_, self.d_, self.k_ = 0, 0, 0

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.d_ = X.shape[1]
        self.m_ = X.shape[0]
        self.classes_ = np.unique(y)
        self.k_ = self.classes_.shape[0]

        self.mu_ = np.zeros((self.k_, self.d_))
        self.cov_ = np.zeros((self.d_, self.d_))
        self.pi_ = np.zeros(self.k_)

        for i, k in enumerate(self.classes_):
            x_k = X[y == k]
            self.mu_[i] = np.mean(x_k)
            self.pi_[i] = x_k.shape[0] / self.m_
            self.cov_ += (x_k - self.mu_[i]).T.dot(x_k - self.mu_[i])

        self.cov_ = self.cov_ / (self.m_ - self.k_)  # unbiased
        self._cov_inv = np.linalg.inv(self.cov_)

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
        a_k = self._cov_inv @ (self.mu_.T)  # DxK
        a_k_T = a_k.T  # KxD
        b = np.log(self.pi_) - 0.5 * np.einsum('kd,kd->k', self.mu_ @ self._cov_inv, self.mu_)  # KxD @ DxD @ KxD

        predictions = np.zeros((self.k_, self.m_))
        for i in range(self.k_):
            predictions[i] = a_k_T[i] @ X.T + b[i]  # m

        k_max = np.argmax(predictions, axis=0)
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

        liklyhoods = np.zeros((self.m_, self.k_))
        Z = np.sqrt((2 * np.pi) ** self.d_ * np.linalg.det(self.cov_))

        for i in range(self.m_):
            for j in range(self.k_):
                x_normal = X[i] - self.mu_[j]
                likly_k = (1 / Z) * np.exp(-0.5 * (x_normal @ self._cov_inv) @ x_normal.T)
                liklyhoods[i, j] = likly_k * self.pi_[j]

        return liklyhoods

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
