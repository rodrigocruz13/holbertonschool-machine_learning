#!/usr/bin/env python3
"""
class RNNCell
"""

import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    function that performs forward propagation for a simple RNN:
    Arguments
    ---------
    - rnn_cell          RNNCell that will be used for the forward prop
    - X                 Data to be used, given as a numpy.ndarray of shape
                        (t, m, i)
                - t     maximum number of time steps
                - m     batch size
                - i     dimensionality of the data
    - h_0               Initial hidden state, given as a numpy.ndarray of
                        shape (m, h)
                - h     dimensionality of the hidden state

    Returns
    -------
    H, Y
        H               numpy.ndarray containing all of the hidden states
        Y               numpy.ndarray containing all of the outputs

    """

    # https://victorzhou.com/blog/intro-to-rnns/

    h_prev = h_0    # shape (m, h)
    # print(np.array([h_0]))
    h_next = np.array([h_0])

    t = X.shape[0]  # maximum number of time steps
    for i in range(t):
        # forward propagation
        h_prev, y = rnn_cell.forward(h_prev, X[i])

        # storage data
        H = np.append(h_next, [h_prev], axis=0)
        Ŷ = np.array([y]) if i == 0 else np.append(Ŷ, [y], axis=0)

    return H, Ŷ
