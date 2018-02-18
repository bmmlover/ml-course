import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler


class LogReg(BaseEstimator):
    def __init__(self, lambda_1=0.0, lambda_2=1.0, gd_type='stochastic',
                 tolerance=1e-4, max_iter=1000, w0=None, alpha=1e-2):
        """
        lambda_1: L1 regularization param
        lambda_2: L2 regularization param
        gd_type: 'full' or 'stochastic'
        tolerance: for stopping gradient descent
        max_iter: maximum number of steps in gradient descent
        w0: np.array of shape (d) - init weights
        alpha: learning rate
        """
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.gd_type = gd_type
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.w0 = w0
        self.alpha = alpha
        self.w = None
        self.loss_history = None
        # self.scaler = StandardScaler()

    def fit(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: self
        """

        self.w = self.w0
        d = X.shape[1]

        if self.w is None:
            self.w = np.random.uniform(-1 / d, 1 / d, d)

        self.loss_history = []

        # X = np.hstack((X, np.ones(X.shape[0])[:, np.newaxis]))

        if self.gd_type == 'stochastic':
            self.loss_history.append(self.calc_loss(X, y))

            for i in range(self.max_iter):
                grad = self.calc_gradient(X[i:i + 1], y[i:i + 1])
                grad_step = self.alpha * grad
                self.w -= grad_step
                self.loss_history.append(self.calc_loss(X, y))

                if numpy.sum(grad_step ** 2) ** 0.5 < self.tolerance:
                    break

        if self.gd_type == 'full':
            self.loss_history.append(self.calc_loss(X, y))

            for i in range(self.max_iter):
                grad_step = self.alpha * self.calc_gradient(X, y)
                self.w -= grad_step
                self.loss_history.append(self.calc_loss(X, y))

                if numpy.sum(grad_step ** 2) ** 0.5 < self.tolerance:
                    break

        return self

    def predict_proba(self, X):
        """
        X: np.array of shape (l, d)
        ---
        output: np.array of shape (l, 2) where
        first column has probabilities of -1
        second column has probabilities of +1
        """
        if self.w is None:
            raise Exception('Not trained yet')

        # X = np.hstack((X, np.ones(X.shape[0])[:, np.newaxis]))

        proba = expit(np.einsum('j,ij->i', self.w, X))
        proba = np.stack((1 - proba, proba), axis=-1)

        return proba

    def calc_gradient(self, X, y):
        """
        X: np.array of shape (l, d) (l can be equal to 1 if stochastic)
        y: np.array of shape (l)
        ---
        output: np.array of shape (d)
        """
        reg = self.lambda_2 * self.w
        main1 = (y[:, np.newaxis] * X)
        main2 = expit(- y * X.dot(self.w))[:, np.newaxis]
        return (-(main1 * main2)).mean(axis=0) + reg

    def calc_loss(self, X, y):
        """
        X: np.array of shape (l, d)
        y: np.array of shape (l)
        ---
        output: float
        """
        l = X.shape[0]
        reg = self.lambda_2 * np.sum(self.w ** 2) / 2
        main = np.logaddexp(0, -y * np.einsum('j,ij->i', self.w, X))
        return np.mean(main) + reg
