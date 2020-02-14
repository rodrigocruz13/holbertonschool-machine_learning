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
        if type(nx) is not int:
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.randn(1, nx)
        #  Draw random samples from a normal dist.
        self.b = 0
        self.A = 0
