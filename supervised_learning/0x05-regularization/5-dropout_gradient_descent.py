#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    creates a tensorflow layer that includes L2 regularization
    Args:
        - prev is a tensor containing the output of the previous layer
        - n is the number of nodes the new layer should contain
        - activation is the activation funct that should be used on the layer
        - lambtha is the L2 regularization paramet
    Returns:
        the output of the new layer
    """

    weights2 = weights.copy()
    m = Y.shape[1]

    for neural_lyr in reversed(range(L)):
        n = neural_lyr + 1
        if (n == L):
            dz = cache["A" + str(n)] - Y
            dw = (np.matmul(cache["A" + str(neural_lyr)], dz.T) / m).T
        else:
            dz1 = np.matmul(weights2["W" + str(n + 1)].T, current_dz)
            dz2 = 1 - cache["A" + str(n)]**2
            dz = dz1 * dz2 * cache['D' + str(n)] / keep_prob
            dw = np.matmul(dz, cache["A" + str(neural_lyr)].T) / m

        db = np.sum(dz, axis=1, keepdims=True) / m
        weights["W" + str(n)] -= (alpha * dw)
        weights["b" + str(n)] -= alpha * db
        current_dz = dz
