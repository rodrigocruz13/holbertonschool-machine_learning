#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np


def variance(X, C):
    """
    Funtion that calculates the total intra-cluster variance for a data set:

    Args:
    - X         numpy.ndarray       Array of shape (n, d) containing the
                                    dataset that will be used for K-means
                                    clustering
        - n     int                 Number of data points
        - d     int                 Number of dimensions for each data point

    - C         numpy.ndarray       Array of shape (k, d) containing the
                                    centroid means for each cluster

        - k     int                 Positive integer containing the number
                                    of clusters
        - d     int                 Number of dimensions for each data point
    Returns:
    - var       float               (total variance), or None on failure
    """

    """
    k = C.shape[0]
    try:
        # Calculate the distance

        n = X.shape[0]
        distances = np.zeros((n, k))
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - C[i], axis=1)

        σ = np.sum(distances) - np.sum(np.min(distances, axis=0))

        return  (σ)/n

    except Exception:
        return None
    """

    if (not isinstance(X, np.ndarray)) or (not isinstance(C, np.ndarray)):
        return None

    if (len(X.shape) != 2) or (len(C.shape) != 2):
        return None

    if (X.size < 1) or (C.size < 1):
        return None

    try:
        # https://paris-swc.github.io/advanced-numpy-lesson/05-kmeans.html

        deltas = X[:, np.newaxis] - C
        dist = np.sqrt(np.sum((deltas) ** 2, axis=2))
        min_dist = np.min(dist, axis=1)
        var = np.sum(min_dist ** 2)
        return var

    except BaseException:
        return None
