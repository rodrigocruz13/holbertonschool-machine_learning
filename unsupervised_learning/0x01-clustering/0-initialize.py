#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np


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

    The cluster centroids should be initialized with a multivariate uniform
    distribution along each dimension in d:
        * The minimum values for the distribution should be the minimum
          values of X along each dimension in d
        * The maximum values for the distribution should be the maximum
          values of X along each dimension in d
    You should use numpy.random.uniform exactly once

    Returns:
    A numpy.ndarray of shape (k, d) containing the initialized centroids
    for each cluster, or None on failure
    """

    d = X.shape[1]

    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None

    if (k < 1):
        return None

    min_ = np.amin(X, axis=0)
    max_ = np.amax(X, axis=0)

    # generate cluster centers (CC)
    try:
        CC = np.random.uniform(low=min_, high=max_, size=(k, d))
        return CC

    except BaseException:
        return None
