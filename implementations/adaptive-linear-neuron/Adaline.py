import numpy as np

class Adaline:
    """
    Perceptron: Linear model classifier
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

    def __init__(self, eta=0.01, n_iter=10, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """
        Train data

        Parameters
        ~~~~~~~~~~
        X : {array-like}, shape = [n_examples, n_features]
            Training vectors, where n_examples is the number of  examples and n_features is the number of features.
        y : array-like, shape = [n_examples]
            Target values.

        Returns
        ~~~~~~~
        self : object
        """
        rand_gen = np.random.RandomState(self.random_state)
        self.weights_ = rand_gen.normal(loc=0.0, scale=0.01,
                                        size=X.shape[1])  # select num of features
        self.bias_ = np.float64(0.0)
        self.losses_ = []

        n = X.shape[0]

        for _ in range(self.n_iter):
            net_input = self.net_input(X)  # wi xi + bias

            result = self.activation(net_input)
            errors = y - result

            # loss function = Mean Squared Error
            # update weights based on the gradient of the loss function respect to weight
            loss_gradient_weight = -(2.0 / n) * X.T.dot(errors)
            self.weights_ -= self.eta * loss_gradient_weight

            # update bias based on the gradient of the loss function respect to bias
            loss_gradient_bias = -(2.0 / n) * errors.sum()
            self.bias_ -= self.eta * loss_gradient_bias

            loss = (errors ** 2).mean()  # calculate mse
            self.losses_.append(loss)

        return self

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)

    def activation(self, X):
        """
        Linear activation function (identity in this case)
        """
        return X

    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self.weights_) + self.bias_