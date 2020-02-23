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
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")

        if layers[0] < 1:
            raise ValueError("layers must be a list of positive integers")

        if (np.where(layers[0] < 0, True, False)):
            raise TypeError("layers must be a list of positive integers")

        self.L = layers[0]
        self.cache = {}
        self.weights = {}

        "layer_size = ls"
        "w=np.random.randn(ls[l],ls[l-1])*np.sqrt(2/ls[l-1])"

        for l in range(len(layers)):
            if not isinstance(layers[l], int):
                raise TypeError('layers must be a list of positive integers')

            if layers[l] <= 0:
                raise TypeError('layers must be a list of positive integers')

            if l == 0:
                he_et_tal = np.random.randn(layers[l], nx) * np.sqrt(2 / nx)

            else:
                he_et_tal = np.random.randn(layers[l], layers[l - 1])
                he_et_tal = he_et_tal * np.sqrt(2 / layers[l - 1])

            self.weights["b" + str(l + 1)] = np.zeros((layers[l], 1))
            self.weights["W" + str(l + 1)] = he_et_tal
