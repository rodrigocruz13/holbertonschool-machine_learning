#!/usr/bin/env python3
"""
Class NeuralNetwork
"""

import numpy as np


class NeuralNetwork:
    """ Class """

    def __init__(self, nx, nodes):
        """
        Initialize NeuralNetwork
        Args:
            - nx: is the number of input features to the neuron
            - Nodes: is the number of nodes found in the hidden layer
        Public attributes:
        - W1: The weights vector for the hidden layer. Upon instantiation,
              it should be initialized using a random normal distribution.
        - b1: b1: The bias for the hidden layer. Upon instantiation,
              it should be initialized with 0’s.
        - A1: The activated output for the hidden layer. Upon instantiation,
              it should be initialized to 0.
        - W2: The weights vector for the neuron.
              It is initialized with a random normal distribution.
        - b2: The bias for the neuron. Upon instantiation.
             It is initialized to 0.
        - A2: The activated output of the neuron (prediction).
            It is initialized to 0.
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        #  Draw random samples from a normal dist.
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        self.__W2 = np.random.randn(1, nodes)
        #  Draw random samples from a normal dist.
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        getter method
        Args:
            - self
        Return:
        - __W1: The value of the proate attribute __W1.
        """
        return self.__W1

    @property
    def A1(self):
        """
        getter method
        Args:
            - self
        Return:
        - __A1: The value of the proate attribute __A1.
        """
        return self.__A1

    @property
    def b1(self):
        """
        getter method
        Args:
            - self
        Return:
        - __b1: The value of the proate attribute __b1.
        """
        return self.__b1

    @property
    def W2(self):
        """
        getter method
        Args:
            - self
        Return:
        - __W2: The value of the proate attribute __W2.
        """
        return self.__W2

    @property
    def A2(self):
        """
        getter method
        Args:
            - self
        Return:
        - __A2: The value of the proate attribute __A2.
        """
        return self.__A2

    @property
    def b2(self):
        """
        getter method
        Args:
            - self
        Return:
        - __b2: The value of the proate attribute __b2.
        """
        return self.__b2
