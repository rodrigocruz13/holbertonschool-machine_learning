#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Function that updates the weights and biases of a neural network using
    gradient descent with L2 regularization:

    Arguments
    ---------
    - Y       : one-hot numpy.ndarray of shape (classes, m) that contains the
                correct labels for the data
                    - classes is the number of classes
                    - m is the number of data points
    - weights : dictionary of the weights and biases of the neural network
    - cache   : dictionary of the outputs of each layer of the neural network
    - alpha   : learning rate
    - lambtha : L2 regularization parameter
    - L       : number of layers of the network

    Note: The NN uses tanh activations on each layer except the last, which
          uses a softmax activation
          The weights and biases of the network should be updated in place
    """

    weights2 = weights.copy()
    m = Y.shape[1]

    for net_ly in reversed(range(L)):

        n = net_ly + 1
        if (n == L):
            dz = cache["A" + str(n)] - Y
            dw = (np.matmul(cache["A" + str(net_ly)], dz.T) / m).T
        else:
            dz1 = np.matmul(weights2["W" + str(n + 1)].T, current_dz)
            dz2 = 1 - cache["A" + str(n)]**2
            dz = dz1 * dz2
            dw = np.matmul(dz, cache["A" + str(net_ly)].T) / m

        dw_reg = dw + (lambtha / m) * weights2["W" + str(n)]
        db = np.sum(dz, axis=1, keepdims=True) / m

        weights["W" + str(n)] -= (alpha * dw_reg)
        weights["b" + str(n)] -= (alpha * db)

        current_dz = dz
