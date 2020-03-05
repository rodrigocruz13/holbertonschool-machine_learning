#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a NN using batch normalization:
    Args:
        Z:      is a numpy.nd array of shape (m, n) that should be normalized
                - m is the number of data points
                - n is the number of features in Z
        gamma:  is a numpy.ndarray of shape (1, n) containing the scales used
                for batch normalization
        beta:   is a numpy.ndarray of shape (1, n) containing the offsets used
                for batch normalization
        epsilon: is a small number used to avoid division by zero

    Returns:
        The the normalized Z matrix
    """

    β = beta
    γ = gamma
    ε = epsilon

    # https://www.youtube.com/watch?v=tNIpEZLv_eg

    # mean
    μ = Z.mean(0)

    # std deviation
    σ = Z.std(0)

    # variance
    σ2 = Z.std(0) ** 2

    # z normalized
    z_normalized = (Z - μ) / ((σ2 + ε) ** (0.5))

    # We dont want all the units to have mean 0 and variance 1
    Ẑ = γ * z_normalized + β

    return Ẑ
