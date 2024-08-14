import numpy as np

class LinearRegressionGD:
    """
    Gradient-Descent Linear Regression
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    Parameters
    ~~~~~~~~~~
    eta : float
        Learning rate, value in range [0,1]
    n_iter : int
        Number of passes over the training data (epochs)
    random_state : int
        Random seed value for initializations

    Attributes
    ~~~~~~~~~~
    weights_ : array-like
        Weights obtained after fitting
    bias_ : Scalar
        Bias unit obtained after fitting
    losses_ : array-like
        MSE loss function in each epoch
    """

    def __init__(self, eta=0.01, n_iter=50, random_state=13):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """Ordinary Least Squares method"""
        rand_gen = np.random.RandomState(self.random_state)
        self.weights_ = rand_gen.normal(loc=0.0, scale=0.01,
                                        size=X.shape[1])  # take num of features
        self.bias_ = np.array([0.0])
        self.losses_ = []

        n = X.shape[0]

        for _ in range(self.n_iter):
            output = self.net_input(X)

            errors = y - output

            # loss function for OLS = Mean Squared Error
            # update weights based on the gradient of the loss function respect to weight
            loss_gradient_weight = -(2.0 / n) * X.T.dot(errors)
            self.weights_ -= self.eta * loss_gradient_weight

            # update bias based on the gradient of the loss function respect to bias
            loss_gradient_bias = -(2.0 / n) * errors.sum()
            self.bias_ -= self.eta * loss_gradient_bias

            loss = (errors ** 2).mean()  # calculate mse
            self.losses_.append(loss)

        return self

    def net_input(self, X):
        return np.dot(X, self.weights_) + self.bias_

    def predict(self, X):
        # Same as Adaline but without the threshold
        return self.net_input(X)