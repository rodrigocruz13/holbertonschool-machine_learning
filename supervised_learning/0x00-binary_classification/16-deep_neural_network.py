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
            - nx: is the number of input features to the neuron
            - layers: is a list representing the number of nodes
                      in each layer of the network
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
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if layers.len == 0:
            raise TypeError("layers must be a list of positive integers")
        if min(layers) < 1:
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(self.L):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                w = np.random.randn(layers[i], nx) * np.sqrt(2 / nx)

            else:
                w = np.random.randn(layers[i], layers[i - 1])
                w = w * np.sqrt(2 / layers[i - 1])

            self.weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            self.weights["W" + str(i + 1)] = w
