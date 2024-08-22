import numpy as np


class Perceptron:
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
    errors_ : array-like
        Number of misclassification errors in each epoch
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
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, yi in zip(X, y):  # loop over features, target
                update = self.eta * (yi - self.predict(xi))
                self.weights_ += update * xi  # update weights array and bias
                self.bias_ += update
                errors += 1 if update != 0 else 0
            self.errors_.append(errors)

        return self

    def predict(self, X):
        """
        Return class label after unit step
        """
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def net_input(self, X):
        """
        Calculate net input
        """
        return np.dot(X, self.weights_) + self.bias_