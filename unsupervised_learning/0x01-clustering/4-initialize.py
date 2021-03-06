#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    Funtion that initializes cluster centroids for K-means:

    Args:
    - X         numpy.ndarray       Array of shape (n, d) containing the
                                    dataset that will be used for K-means
                                    clustering
        - n     int                 Number of data points
        - d     int                 Number of dimensions for each data point

    - k         int                 Positive integer containing the number
                                    of clusters

    Returns:
    pi, m, S,               if sucess
    None, None, None        on failure
        - pi        numpy.ndarray   Array of shape (k,) containing the priors
                                    for each cluster, initialized evenly
        - m         numpy.ndarray   Array of shape (k, d) containing the
                                    centroid means for each cluster,
                                    initialized with K-means
        - S         numpy.ndarray   Array of shape (k, d, d) containing the
                                    covariance matrices for each cluster,
                                    initialized as identity matrices
    """

    try:
        if not isinstance(X, np.ndarray) or not isinstance(k, int):
            return None, None, None

        if (X.ndim != 2):
            return None, None, None

        if (k < 1) or (X.shape[0] <= k):
            return None, None, None

        # generate priors for each cluster, initialized evenly
        d = X.shape[1]

        pi = np.array([1 / k] * k).reshape((k, ))
        m = kmeans(X, k)[0]
        S = np.array([np.identity(d, dtype=float)] * k)

        if (m is None):
            return None, None, None

        return pi, m, S

    except BaseException:
        return None, None, None
