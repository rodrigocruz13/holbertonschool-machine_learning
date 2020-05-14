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

    # calculate covariance matrix of centered matrix
    vh = np.linalg.svd(X)[2]
    s = np.linalg.svd(X)[1]

    accum_var = np.cumsum(s) / np.sum(s)
    results = np.argwhere(accum_var >= var)[0, 0]

    return vh[:results + 1].T
