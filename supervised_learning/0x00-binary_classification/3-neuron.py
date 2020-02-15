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
        - X is a numpy.ndarray with shape (nx, m) that contains the input data
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
        cost = j * np.sum(np.multiply(Y, np.log(A)) +
                          np.multiply(1 - Y, np.log(1 - A)))
        return cost
