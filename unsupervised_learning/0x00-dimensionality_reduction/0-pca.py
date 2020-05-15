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

    # 1. calculate the single value decomposition
    singular_values = np.linalg.svd(X)[1]
    vectors_horizontal = np.linalg.svd(X)[2]

    # The singular values (s), sorted in non-increasing order. Of shape (K,),
    # with K = min(M, N).
    num = np.cumsum(singular_values)
    denom = np.sum(singular_values)
    accum_var =  num / denom

    # filter according specs of var
    results = np.argwhere(accum_var >= var)
    res = results[0, 0] + 1
    weights = vectors_horizontal[ : res].T

    return weights