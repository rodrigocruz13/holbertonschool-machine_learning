#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def normalize(X, m, s):
    """
    Normalizes (standardizes) a matrix X:

    Args:
        X numpy.ndarray of shape (m, nx) to normalize
            - m is the number of data points
            - nx is the number of features
        m is a numpy.ndarray of shape (nx,) that contains the mean
        of all features of X
        s is a numpy.ndarray of shape (nx,) that contains the standard
        deviation of all features of X
    Returns:
        The normalized X matrix
    """

    # mean
    Ẋ = np.mean(X, axis=0)

    # Standard deviation
    σ = np.std(X, axis=0)

    return (Ẋ, σ)
