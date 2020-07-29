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

    #    - vh { (…, N, N), (…, K, N) } array
    # Unitary array(s). The first a.ndim - 2 dimensions have the same size as
    # those of the input a. The size of the last two dimensions depends on the
    # value of full_matrices. Only returned when compute_uv is True.
    vh = np.linalg.svd(X_normal)[2]

    # 3. filter according ndim
    Weights_r = vh[: ndim].T

    # line 20 of 0-main.py
    T = np.matmul(X_normal, Weights_r)

    return T
