#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np


def pca(X, ndim):
    """
    Function that performs PCA on a dataset:

    Args:
    - X         numpy.ndarray   Array of shape (n, d) where:
                - n     (int)       number of data points
                - d     (int)       number of dims in each point
                Note: All dims have a mean of 0 across all data points
    - ndim      (int)           New dimensionality of the transformed X

    Returns:    (The weights matrix that maintains var fraction of X‘s
                original variance)
    T:          numpy.ndarray of shape (d, ndim) containing the
                transformed version of X
    """

    # https://bit.ly/2zH844p

    # 1. Normalize
    normal = np.mean(X, axis=0)
    X_normal = X - normal

    # 2. calculate the single value decomposition
    vectors_horizontal = np.linalg.svd(X_normal)[2]

    # 3. filter according ndim
    res = vectors_horizontal[: ndim].T
    T = X_normal @ res

    return T
