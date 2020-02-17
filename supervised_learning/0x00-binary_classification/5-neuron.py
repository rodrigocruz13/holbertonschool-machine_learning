#!/usr/bin/env python3
"""
Class Neuron
"""

import numpy as np


class Neuron:
    """ Class """

    def __init__(self, nx):
        """
        Initialize Neuron
        Args:
            - nx: nx is the number of input features to the neuron
        Public attributes:
        - W: The weights vector for the neuron.
              It is initialized with a random normal distribution.
        - b: The bias for the neuron. Upon instantiation.
             It is initialized to 0.
        - A: The activated output of the neuron (prediction).
            It is initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.randn(1, nx)
        #  Draw random samples from a normal dist.
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        getter method
        Args:
            - self
        Return:
        - __W: The value of the attribute __W.
        """
        return self.__W

    @property
    def A(self):
        """
        getter method
        Args:
            - self
        Return:
        - __A: The value of the attribute __A.
        """
        return self.__A

    @property
    def b(self):
        """
        getter method
        Args:
            - self
        Return:
        - __b: The value of the attribute __b.
        """
        return self.__b

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neuron
        Updates the private attribute __A
        Uses a sigmoid activation function
        Arguments:
          - X is a numpy.ndarray with shape (nx, m) & contains the input data
          - nx (int) is the number of input features to the neuron
          - m (int) is the number of examples
        Return:
        - __A: The value of the attribute __A.
        """
        # z = w.X + b
        z = np.matmul(self.W, X) + self.b
        forward_prop = 1 / (1 + np.exp(-1 * z))
        self.__A = forward_prop
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
        - Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        - A is a numpy.ndarray with shape (1, m) containing the activated
        output of the neuron for each example
        Return:
        - cost: the cost
        Answer from: https://bit.ly/37x9YzM
        """
        m = Y.shape[1]
        j = - (1 / m)

        Ä = 1.0000001 - A
        Ÿ = 1 - Y
        cost = j * np.sum(np.multiply(Y, np.log(A)) +
                          np.multiply(Ÿ, np.log(Ä)))
        return cost

    def evaluate(self, X, Y):
        """
        Calculates the cost of the model using logistic regression
        Arguments:
         - X is a numpy.ndarray with shape (nx, m) & contains the input data

        - Y is a numpy.ndarray with shape (1, m) that contains the correct
            labels for the input data
        Return:
        - Prediction: The prediction should be a numpy.ndarray with shape
                      (1, m) containing the predicted labels for each
                      example. The label values should be 1 if the output
                      of the network is >= 0.5 and 0 otherwise
        - cost: the cost
        Answer from: https://bit.ly/37x9YzM
        """

        # Generate forward propagation.
        # This creates the value of each activation
        predictions = self.forward_prop(X)

        # Calculate cost
        cost = self.cost(Y, self.__A)

        # Generate a matrix of size (1, m) and generate labels
        # 1 if the output of the network is >= 0.5 and 0 otherwise

        # You are not allowed to use any loops (for, while, etc.)

        # labels = self.__A.copy()
        # for i in range(len(self.__A)):
        #    for j in range(len(self.__A[0])):
        #        if (self.__A[i][j] >= 0.5):
        #            labels[i][j] = 1
        #        else:
        #            labels[i][j] = 0
        # return (labels, cost)

        labels = np.where(predictions < 0.5, 0, 1)
        return (labels, cost)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neuron
        and updates the private attributes __W and __b

        Arguments:
        - X (numpy.ndarray) with shape (nx, m) & contains the input data
        - Y (numpy.ndarray) with shape (1, m) that contains the correct
            labels for the input data
        - A (numpy.ndarray) with shape (1, m) containing the activated
            output of the neuron for each example
        - alpha: is the learning rate.

        Return:
        - Prediction: The prediction should be a numpy.ndarray with shape
                      (1, m) containing the predicted labels for each
                      example. The label values should be 1 if the output
                      of the network is >= 0.5 and 0 otherwise
        - cost: the cost
        Answer from: - https://bit.ly/37vaP3S

        """

        α = alpha
        m = 1 / len(X[0])

        # z = w1X1 + w2X2 + b
        # Derivative z: https://www.youtube.com/watch?v=z_xiwjEdAC4  
        dz = A - Y

        # Derivative respect to weight
        dw = np.matmul(X, (dz).T) * m

        # Derivative respect to bias
        db = np.sum(dz) * m

        # w: = α * dw, where ": = means actualization"
        self.__W = self.__W - (α * dw)

        # b: = α * db, where ": = means actualization"
        self.__b = self.__b - (α * db)
