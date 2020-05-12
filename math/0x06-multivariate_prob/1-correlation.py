#!/usr/bin/env python3
""" module """

import numpy as np


def correlation(C):
    """
    Function that calculates a correlation matrix:

    Args:
        - C:        numpy.ndarray   Array of shape (d, d) containing
                    a covariance matrix:

        If C is not a 2D numpy.ndarray, raise a TypeError with the message:
        C must be a numpy.ndarray
        If C does not have shape (d, d), raise a ValueError with the message
        C must be a 2D square matrix

        If n is less than 2, raise a ValueError with the message
        X must contain multiple data points

    Returns:
        - correlation   numpy ndarray   Matrix of shape (d, d) with its corr
    """
    if (isinstance(C, type(None))):
        raise TypeError('C must be a numpy.ndarray')

    if (not isinstance(C, np.ndarray)):
        raise TypeError('C must be a numpy.ndarray')

    if (len(C.shape) != 2):
        raise ValueError("C must be a 2D square matrix")

    if (C.shape[0] != C.shape[1]):
        raise ValueError("C must be a 2D square matrix")

    # source: https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b

    v = np.sqrt(np.diag(C))
    outer_v = np.outer(v, v)
    correlation = C / outer_v

    return correlation
