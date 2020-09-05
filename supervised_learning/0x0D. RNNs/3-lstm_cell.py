#!/usr/bin/env python3

"""
class LSTMcell
"""

import numpy as np


class LSTMCell:
    """
        class LSTMcell that represents a LSTM unit
    """

    def __init__(self, i, h, o):
        """
        Constructor

        Args:
            i:      i is the dimensionality of the data
            h:      is the dimensionality of the hidden state
            o:      is the dimensionality of the outputs

        Attributes that represent the weights and biases of the cell:
            Wf:     weights for the forget gate
            Wu:     weights for the update gate
            Wc:     weights for the intermediate cell state
            Wo:     weights for the are for the output gate
            Wy:     weights for the output

            bf:     bias for the forget gate
            bu:     bias for the update gate
            bc:     bias for the intermediate cell state
            bo:     bias for the are for the output gate
            by:     bias for the output

        The weights should be initialized using a random normal distribution
        in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """

        # initializating Weights in order
        self.Wf = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wu = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wc = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wo = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wy = np.random.normal(size=(h, o))  # size = (15, 5)

        # initializating bias in order
        self.bf = np.zeros(shape=(1, h))
        self.bu = np.zeros(shape=(1, h))
        self.bc = np.zeros(shape=(1, h))
        self.bo = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        public instance method that performs forward propagation 4 1 time step

        Arguments:
        ---------
        h_prev  numpy.ndarray   array of shape (m, h) containing the previous
                                hidden state
                        m       is the batche size for the data
                        h       is the dimensionality of the hidden state
        c_prev  numpy.ndarray   array of shape (m, h) containing the previous
                                cell state
        The output of the cell should use a softmax activation function

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

        # gate u:
        u = np.dot(x, self.Wu) + self.bu
        # activating usng sigmoid
        u = 1 / (1 + np.exp(-u))

        # gate f:
        f = np.dot(x, self.Wf) + self.bf
        # activating usng sigmoid
        f = 1 / (1 + np.exp(-f))

        # gate o:
        o = np.dot(x, self.Wo) + self.bo
        # activating using sigmoid
        o = 1 / (1 + np.exp(-o))

        c = np.tanh(np.dot(x, self.Wc) + self.bc)
        c_t = u * c + f * c_prev
        h_t = o * np.tanh(c_t)

        # ŷ = Wₕᵧ · hₜ + bᵧ
        ŷ = np.dot(h_t, self.Wy) + self.by

        # Activating using softmax
        y = (np.exp(ŷ) / np.sum(np.exp(ŷ), axis=1, keepdims=True))

        return h_t, c_t, y
