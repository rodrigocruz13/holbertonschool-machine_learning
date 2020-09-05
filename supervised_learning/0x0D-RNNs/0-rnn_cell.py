#!/usr/bin/env python3

"""
class RNNCell
"""

import numpy as np


class RNNCell:
    """
        class RNNCell that represents a cell of a simple RNN
    """

    def __init__(self, i, h, o):
        """
        Constructor

        Args:
            i:      i is the dimensionality of the data
            h:      is the dimensionality of the hidden state
            o:      is the dimensionality of the outputs
        Attributes that represent the weights and biases of the cell:
            Wh:     weights for the concatenated hidden state and input data
            Wy:     weights for the output
            bh:     bias for the concatenated hidden state and input data
            by:     bias for the output
        The weights should be initialized using a random normal distribution
        in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """

        # http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-model-rnn-with-python-numpy-and-theano/

        self.Wh = np.random.normal(size=(h + i, h))  # size = (25, 15)
        self.Wy = np.random.normal(size=(h, o))  # size = (15, 5)
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

        # hₜ = tanh(Wₓₕ · xₜ + Wₕₕ · hₜ₋₁ + bₕ)
        x = np.concatenate((h_prev, x_t), axis=1)
        h_t = np.tanh(np.dot(x, self.Wh) + self.bh)

        # ŷ = Wₕᵧ · hₜ + bᵧ
        ŷ = np.dot(h_t, self.Wy) + self.by

        # Activating using softmax
        y = (np.exp(ŷ) / np.sum(np.exp(ŷ), axis=1, keepdims=True))

        return h_t, y
