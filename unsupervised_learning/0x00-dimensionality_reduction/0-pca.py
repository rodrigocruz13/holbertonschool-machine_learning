#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np


def pca(X, var=0.95):
    """
    Function that performs PCA on a dataset:

    Args:
    - X         numpy.ndarray   Array of shape (n, d) where:
                - n     (int)       number of data points
                - d     (int)       number of dims in each point
                Note: All dims have a mean of 0 across all data points
    - var       (float)         Fraction of the variance that the PCA
                                transformation should maintain

    Returns:    (The weights matrix that maintains var fraction of X‘s
                original variance)
    W:          numpy.ndarray of shape (d, nd)
                - nd                new dimensionality of the transformed X

    """

    # symbols: ⊤, ⅀, σ²³

    # 1. calculate the single value decomposition
    # s(…, K) array
    # Vector(s) with the singular values, within each vector sorted in
    # descending order. The first a.ndim - 2 dimensions have the same size as
    # those of the input a.
    # { (…, N, N), (…, K, N) } array
    # Unitary array(s). The first a.ndim - 2 dimensions have the same size as
    # those of the input a. The size of the last two dimensions depends on the
    # value of full_matrices. Only returned when compute_uv is True.

    s = np.linalg.svd(X)[1]
    vh = np.linalg.svd(X)[2]

    num = np.cumsum(s)
    denom = np.sum(s)
    accum_var = num / denom

    # filter according specs of var
    num_truncated_results = np.argwhere(accum_var >= var)
    num_truncated_results = num_truncated_results[0, 0] + 1
    weights = vh[: num_truncated_results].T
    # print("weights ", weights)

    return weights
