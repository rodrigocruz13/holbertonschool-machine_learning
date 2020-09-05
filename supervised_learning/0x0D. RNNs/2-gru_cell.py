#!/usr/bin/env python3

"""
class GRUCell
"""

import numpy as np


class GRUCell:
    """
        class GRUCell that represents a gated recurrent unit
    """

    def __init__(self, i, h, o):
        """
        Constructor

        Args:
            i:      i is the dimensionality of the data
            h:      is the dimensionality of the hidden state
            o:      is the dimensionality of the outputs

        Attributes that represent the weights and biases of the cell:
            Wz:     weights for the concatenated update gate
            Wr:     weights for the reset gate
            Wh:     weights for the concatenated hidden state and input data
            Wy:     weights for the output

            bz:     bias for the update gate
            Wz:     weights for the reset gate
            bh:     bias for the concatenated hidden state and input data
            by:     bias for the output
        The weights should be initialized using a random normal distribution
        in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """

        # initializating Weights in order
        self.Wz = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wr = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wh = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wy = np.random.normal(size=(h, o))  # size = (15, 5)

        # initializating bias in order
        self.bz = np.zeros(shape=(1, h))
        self.br = np.zeros(shape=(1, h))
        self.bh = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        public instance method that performs forward propagation 4 1 time step

        Arguments:
        ---------
        h_prev  numpy.ndarray   array of shape (m, h) containing the previous
                                hidden state
                        m       is the batche size for the data
                        h       is the dimensionality of the hidden state
        x_t     numpy.ndarray   array of shape (m, i) that contains the data
                                input for the cell
        The output of the cell should use a softmax activation function
        Returns:
        --------
        h_t, y
            h_t is the next hidden state
            y is the output of the cell
        """
        # https://victorzhou.com/blog/intro-to-rnns/

        x = np.concatenate((h_prev, x_t), axis=1)

        # gate z:
        z = np.dot(x, self.Wz) + self.bz
        # activating usng sigmoid
        z = 1 / (1 + np.exp(-z))

        # gate r:
        r = np.dot(x, self.Wr) + self.br
        # activating using sigmoid
        r = 1 / (1 + np.exp(-r))

        x = np.concatenate((r * h_prev, x_t), axis=1)
        h = np.tanh(np.dot(x, self.Wh) + self.bh)
        h_t = z * h + (1 - z) * h_prev

        # ŷ = Wₕᵧ · hₜ + bᵧ
        ŷ = np.dot(h_t, self.Wy) + self.by

        # Activating using softmax
        y = (np.exp(ŷ) / np.sum(np.exp(ŷ), axis=1, keepdims=True))

        return h_t, y
