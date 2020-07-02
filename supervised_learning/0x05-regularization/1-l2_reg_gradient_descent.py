#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    calculates the cost of a neural network with L2 regularization
    Args:
        - cost is the cost of the network without L2 regularization
        - lambtha is the regularization parameter
        - weights is a dictionary of the weights and biases (numpy.ndarrays)
          of the neural network
        - L is the number of layers in the neural network
        - m is the number of data points used
    Returns:
         the cost of the network accounting for L2 regularization
    """

    # from: https://bit.ly/2IAtnFN
    # Cost function = Loss + (λ / 2 * m) *  Σ | w | ^ 2

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

        weights["W" + str(n)] = weights2["W" + str(n)] - (alpha * dw_reg)
        weights["b" + str(n)] = weights2["b" + str(n)] - (alpha * db)

        current_dz = dz
