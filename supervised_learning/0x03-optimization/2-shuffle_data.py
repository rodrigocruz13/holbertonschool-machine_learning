#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way:

    Args:
        X numpy.ndarray of shape (m, nx) to shuffle
            - m is the number of data points
            - nx is the number of features
        Y is the second numpy.ndarray of shape (m, ny)
        to shuffle
            - m is the same number of data points as in X
            - ny is the number of features in Y
    Returns:
        The shuffled X and Y matrices
    """

    # Answer # 2. stackoverflow. https://bit.ly/2TshCq6
    # Using numPy's array indexing:

    assert len(X) == len(Y)
    p = np.random.permutation(len(X))
    return (X[p], Y[p])
