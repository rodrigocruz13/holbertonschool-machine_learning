#!/usr/bin/env python3
"""
Class DeepNeuralNetwork
"""

import numpy as np


class DeepNeuralNetwork:
    """ Class """

    def __init__(self, nx, layers):
        """
        Initialize NeuralNetwork
        Args:
            - nx: nx is the number of input features
            - Layers: is the number of nodes found in the hidden layer
        Public attributes:
            - L: The number of layers in the neural network.
            - cache: A dictionary to hold all intermediary values of the
            network. Upon instantiation, it should be set to an empty dict.
            - weights: A dict to hold all weights and biased of the network.
            Upon instantiation:
            - The weights of the network should be initialized with He et al.
            method and saved in the weights dictionary using the key W{l}
            where {l}is the hidden layer the weight belongs to
            - The biases of the network should be initialized to 0’s and
            saved in the weights dictionary using the key b{l} where {l}
            is the hidden layer the bias belongs to
        """

        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')

        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')

        if len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')

        # Privatizing attributes

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for lay in range(len(layers)):
            if not isinstance(layers[lay], int) or layers[lay] <= 0:
                raise TypeError('layers must be a list of positive integers')

            self.weights["b" + str(lay + 1)] = np.zeros((layers[lay], 1))

            if lay == 0:
                sq = np.sqrt(2 / nx)
                he_et_al = np.random.randn(layers[lay], nx) * sq
                self.weights["W" + str(lay + 1)] = he_et_al

            else:
                sq = np.sqrt(2 / layers[lay - 1])
                he_et_al = np.random.randn(layers[lay], layers[lay - 1]) * sq
                self.weights["W" + str(lay + 1)] = he_et_al

    @property
    def L(self):
        """
        getter method
        Args:
            - self
        Return:
        - __L: NUmber of layers .
        """
        return self.__L

    @property
    def cache(self):
        """
        getter method
        Args:
            - self
        Return:
        - __cache: (dict) Has al intermediaty values of the network.
        """
        return self.__cache

    @property
    def weights(self):
        """
        getter method
        Args:
            - self
        Return:
        - __weights: (dict) Has al the weights and bias of the network.
        """
        return self.__weights
