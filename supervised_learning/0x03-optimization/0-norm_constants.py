#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def normalization_constants(X):
    """
    calculates the normalization (standardization) constants of a matrix:
    Args:
        X numpy.ndarray of shape (m, nx) to normalize
            - m is the number of data points
            - nx is the number of features
    Returns:
        Mean and standard deviation of each feature, respectively
    """

    # mean
    Ẋ = np.mean(X, axis=0)

    # Standard deviation
    σ = np.std(X, axis=0)

    return (Ẋ, σ)
