import numpy as np
from scipy.special import expit as sigmoid

class NeuralNetMLP:

    def __init__(self, n_features, n_hidden, n_classes, random_state=13):
        """
        Initialize the neural network with given parameters.

        :param n_features: Number of input features. Determines the size of the input layer.
        :param n_hidden: Number of neurons in the hidden layer. Determines the size of the hidden layer.
        :param n_classes: Number of output classes. Determines the size of the output layer.
        :param random_state: Seed for the random number generator to ensure reproducibility of results.
        """

        super().__init__()

        self.n_classes = n_classes

        # Hidden layer
        rand_gen = np.random.RandomState(random_state)

        # size: there’s a weight for each connection from every input feature to every hidden neuron
        self.weight_h = rand_gen.normal(loc=0., scale=1., size=(n_hidden, n_features))
        self.bias_h = np.zeros(n_hidden)

        # Output layer

        # size: there’s a weight for each connection from every hidden neuron to each output class
        self.weight_out = rand_gen.normal(loc=0., scale=1., size=(n_classes, n_hidden))
        self.bias_out = np.zeros(n_classes)

    #
    # All the code from now on is descriptively commented for understanding and bearability
    #

    def forward(self, x):
        """
        Forward pass: Compute the activations for the hidden and output layers given the input data.

        :param x: Input data of shape [n_examples, n_features]

        :return:
        - a_h: Hidden layer activations of shape [n_examples, n_hidden]
        - a_out: Output layer activations (class probabilities) of shape [n_examples, n_classes]
        """

        # Hidden layer: Compute linear combination of input features using the hidden layer's weights and biases,
        #               then apply sigmoid function to produce hidden layer activations

        ## x dims: [n_examples x n_features]
        ## w.T dims: [n_features x n_hidden]
        ## dot result, z_h: [n_examples x n_hidden], + b
        z_h = np.dot(x, self.weight_h.T) + self.bias_h
        a_h = sigmoid(z_h)
        # a_h is used to optimize the weights and biases of the hidden and output layers

        # Output layer: compute linear combination of hidden layer activations using the output layer's weights and biases,
        #               then apply sigmoid function to produce output layer activations (class probabilities)

        ## a_h dims: [n_examples x n_hidden]
        ## w.T dims: [n_hidden x n_classes]
        ## dot res, z_out: [n_examples x n_classes], + b
        z_out = np.dot(a_h, self.weight_out.T) + self.bias_out
        a_out = sigmoid(z_out)
        # a_out provides class probabilities

        return a_h, a_out


    def backward(self, x, a_h, a_out, y):
        """
        Backpropagate: Calculate the gradients of the loss with respect to the weight and bias parameters,
                       which are then used for the gradient descent updates.

        :param x: Input data of shape [n_examples, n_features].
        :param a_h: Hidden layer activations of shape [n_examples, n_hidden].
        :param a_out: Output layer activations (predictions) of shape [n_examples, n_classes].
        :param y: True labels of shape [n_examples].

        :return: Gradients for output weights, output biases, hidden weights, and hidden biases.
        """

        ### ### Output layer weights ### ###

        # Convert labels to one-hot encoded format
        y_onehot = self._one_hot_encode(y, self.n_classes)

        # Part 1: dLoss/dOutWeights = dL/dWo
        ## Compute the gradient of the loss with respect to output layer weights
        ## DeltaOut = dLoss/dOutAct * dOutAct/dOutNet

        # Gradient of the loss w.r.t. output activations
        d_loss__d_a_out = 2.0 * (a_out - y_onehot) / len(y)

        # Gradient of the output activation w.r.t. the weighted input (sigmoid derivative)
        d_a_out__d_z_out = a_out * (1.0 - a_out)

        # Delta for output layer, used to propagate error backward
        delta_out = d_loss__d_a_out * d_a_out__d_z_out

        # Gradient for output weights and biases

        # The gradient of the output layer's weighted input w.r.t. the weights
        d_z_out__dw_out = a_h

        # Compute the gradient of the loss w.r.t. output layer weights
        d_loss__dw_out = np.dot(delta_out.T, d_z_out__dw_out)

        # Compute the gradient of the loss w.r.t. output layer biases
        d_loss__db_out = np.sum(delta_out, axis=0)

        ### ###  Hidden layer weights ### ###

        # Part 2: dLoss/dHiddenWeights = dL/dWh
        ## Compute the gradient of the loss w.r.t. hidden layer weights

        # Propagate the error back through the output layer to the hidden layer
        d_z_out__a_h = self.weight_out

        # Gradient of the loss w.r.t. hidden layer activations
        d_loss__a_h = np.dot(delta_out, d_z_out__a_h)

        # Gradient of hidden activations w.r.t. the weighted input (sigmoid derivative)
        d_a_h__d_z_h = a_h * (1.0 - a_h)

        # Compute the gradient of the loss w.r.t. hidden layer weights
        d_loss__d_w_h = np.dot((d_loss__a_h * d_a_h__d_z_h).T, x)

        # Compute the gradient of the loss w.r.t. hidden layer biases
        d_loss__d_b_h = np.sum((d_loss__a_h * d_a_h__d_z_h), axis=0)

        return d_loss__dw_out, d_loss__db_out, d_loss__d_w_h, d_loss__d_b_h


    def _one_hot_encode(self, y, n_classes):
        encoded = np.zeros((len(y), n_classes))
        for i, yi in enumerate(y):
            encoded[i, yi] = 1
        return encoded


