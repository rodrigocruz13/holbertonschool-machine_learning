#!/usr/bin/env python3
"""
Class Neuron
"""

import numpy as np


class Neuron:
    """ Class """

    def __init__(self, nx):
        """
        Initialize the Neuron class

        Arguments
        ---------
        - nx   : number of input features to the neuron

        Return
        ------
        Public attributes:
        - W    : The weights vector for the neuron.
                 It is initialized with a random normal distribution.
        - b    : The bias for the neuron. Upon instantiation.
                 It is initialized to 0.
        - A    : The activated output of the neuron (prediction).
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
        Calculates the forward propagation of the neuron. It also updates the
        private attribute __A by using a sigmoid activation function

        Arguments
        ---------

        - X     : numpy.ndarray
                  Array with shape (nx, m) that contains the input data
             nx : int
                  number of input features to the neuron
             m  : int
                  number of examples
        Return
        ------
        - __A   : float
                  Value of the attribute __A.
        """
        # z = w.X + b
        z = np.matmul(self.W, X) + self.b
        forward_prop = 1 / (1 + np.exp(-1 * z))
        self.__A = forward_prop
        return self.__A

    def cost(self, Y, A):
        """
        Calculates the cost of the model using logistic regression
        Arguments
        ---------
        - Y   : numpy.ndarray
                Array with shape (1, m) with the correct labels for the inputs
        - A   : numpy.ndarray
                Array with shape (1, m) containing the activated output of the
                neuron for each example
        Return:
        - cost: float
                the cost of the model
        """

        # Answer from: https://bit.ly/37x9YzM  - Compute cost
        m = Y.shape[1]
        j = - (1 / m)

        Â = 1.0000001 - A
        Ŷ = 1 - Y
        log_A = np.log(A)
        log_Â = np.log(Â)
        cost = j * np.sum(np.multiply(Y, log_A) + np.multiply(Ŷ, log_Â))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neuron’s predictions

        Arguments

        - X          : numpy.ndarray
                       Array with shape (nx, m) that contains the input data
               nx    : int
                       number of input features to the neuron
               m     : int
                       number of examples
        - Y          : numpy.ndarray
                       Array with shape (1, m) that contains the correct
                       labels for the input data

        Returns
        neuron’s prediction (labels) and the cost of the network, respectively
        - prediction : numpy.ndarray
                       Array with shape (1, m) containing the predicted labels
                       for each example
                       - The label values should be 1 if the output of the
                         network is >= 0.5 and 0 otherwise
        - costs      : float
                       the cost of the model
        """

        # Generate the value of each activation
        predictions = self.forward_prop(X)

        # Calculate cost
        cost = self.cost(Y, self.__A)
        labels = np.where(predictions < 0.5, 0, 1)

        return (labels, cost)
