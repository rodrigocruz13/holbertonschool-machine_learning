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

    # calculate the mean of each column
    # M = mean(A.T, axis=1)
    # print(M)
    # center columns by subtracting column means
    # C = A - M
    # print(C)
    # calculate covariance matrix of centered matrix
    V = np.cov(X.T)
    # print(V)

    # eigendecomposition of covariance matrix
    vectors = np.linalg.eig(V)[1]

    # project data
    P = np.array(vectors.T.dot(X.T))

    return P
