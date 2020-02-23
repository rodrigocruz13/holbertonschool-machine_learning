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

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network
        Updates the private attributes __cache
        The activated outputs of each layer should be saved in the __cache
        dictionary using the key A{l} where {l} is the hidden layer the
        activated output belongs to
        The neurons should use a sigmoid activation function


        Arguments:
        - X is a numpy.ndarray with shape (nx, m) that contains the input data
          - nx (int) is the number of input features to the neuron
          - m (int) is the number of examples
        Return:
        - The output of the neural network and the cache, respectively
        """

        self.__cache["A0"] = X

        for layer in range(self.__L):

            weights = self.__weights["W" + str(layer + 1)]
            a_ = self.__cache["A" + str(layer)]
            b = self.__weights["b" + str(layer + 1)]

            # z1 = w . X1 + b1
            z = np.matmul(weights, a_) + b

            # sigmoid function
            forward_prop = 1 / (1 + np.exp(-1 * z))

            # updating cache
            self.__cache["A" + str(layer + 1)] = forward_prop

        return self.__cache["A" + str(self.__L)], self.__cache
