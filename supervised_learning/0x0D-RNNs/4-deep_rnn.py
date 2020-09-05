#!/usr/bin/env python3

"""
class LSTMcell
"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Function that performs forward propagation for a deep RNN:

    Arguments:
    ---------
    - rnn_cells list                list of RNNCell instances of length l that
                                    will be used for the forward propagation
                l                   number of layers
    - X         numpy.ndarray       array of shape (t, m, i) with the data
                                    to be used
                t                   maximum number of time steps
                m                   batch size
                i                   dimensionality of the data

    - h_0       numpy.ndarray       array of the initial hidden state, with
                                    shape (l, m, h)
                h                   dimensionality of the hidden state
    Returns
    -------
    H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs
    """

    h_prev = h_0
    H = np.array(([h_prev]))
    H = np.repeat(H, X.shape[0] + 1, axis=0)

    for i in range(X.shape[0]):
        for a_layer, cell in enumerate(rnn_cells):

            # forwarding
            parameter = X[i] if a_layer == 0 else h_prev
            h_prev, y = cell.forward(H[i, a_layer], parameter)

            # update the hidden states
            H[i + 1, a_layer] = h_prev

            # update all the outputs
            if (i != 0):
                Y[i] = y

            else:
                Y = np.array([y])
                Y = np.repeat(Y, X.shape[0], axis=0)

    return H, Y
