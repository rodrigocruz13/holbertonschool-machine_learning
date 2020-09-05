#!/usr/bin/env python3

"""
class LSTMcell
"""

import numpy as np


class BidirectionalCell:
    """
        class BidirectionalCell that represents a bidirectional cell of an RNN
    """

    def __init__(self, i, h, o):
        """
        Constructor

        Args:
            i:      i is the dimensionality of the data
            h:      is the dimensionality of the hidden state
            o:      is the dimensionality of the outputs

        Attributes that represent the weights and biases of the cell:
            Whf:    weights for the hidden states in the forward direction
            Whb:    weights for the hidden states in the backward direction
            Wy:     weights for the output

            bhf:    bias for the hidden states in the forward direction
            bhb:    bias for the hidden states in the backward direction
            by:     bias for the output

        The weights should be initialized using a random normal distribution
        in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """

        # initializating Weights in order
        self.Whf = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Whb = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wy = np.random.normal(size=(h + h, o))  # size = (30, 5)

        # initializating bias in order
        self.bhf = np.zeros(shape=(1, h))
        self.bhb = np.zeros(shape=(1, h))
        self.by = np.zeros(shape=(1, o))

    def forward(self, h_prev, x_t):
        """
        public instance method that calculates the hidden state in the forward
        direction for one time step

        Arguments:
        ---------
        h_prev  numpy.ndarray   array of shape (m, h) containing the previous
                                hidden state
                        m       is the batche size for the data
                        h       is the dimensionality of the hidden state
        The output of the cell should use a softmax activation function

        x_t     numpy.ndarray   array of shape (m, i) that contains the data
                                input for the cell
                        i:      i is the dimensionality of the data

        The output of the cell should use a softmax activation function

        Returns:
        --------
        h_t
            h_t is the next hidden state
        """

        # https://victorzhou.com/blog/intro-to-rnns/

        x = np.concatenate((h_prev, x_t), axis=1)
        h_t = np.tanh(np.dot(x, self.Whf) + self.bhf)

        return h_t

    def backward(self, h_next, x_t):
        """
        public instance method that calculates the hidden state in the backward
        direction for one time step

        Arguments:
        ---------
        h_next  numpy.ndarray   array of shape (m, h) containing the next
                                hidden state
                        m       is the batche size for the data
                        h       is the dimensionality of the hidden state
        The output of the cell should use a softmax activation function

        x_t     numpy.ndarray   array of shape (m, i) that contains the data
                                input for the cell
                        i:      i is the dimensionality of the data

        The output of the cell should use a softmax activation function

        Returns:
        --------
        h_pev
            h_pev is the previous hidden state
        """

        x = np.concatenate((h_next, x_t), axis=1)
        h_pev = np.tanh(np.dot(x, self.Whb) + self.bhb)

        return h_pev
