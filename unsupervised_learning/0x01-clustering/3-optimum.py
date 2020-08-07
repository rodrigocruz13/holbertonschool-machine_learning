#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Funtion that tests for the optimum number of clusters by variance

    Args:
    - X         numpy.ndarray       Array of shape (n, d) containing the
                                    dataset that will be used for K-means
                                    clustering
        - n     int                 Number of data points
        - d     int                 Number of dimensions for each data point

    - kmin      int                 Positive integer containing the minimum
                                    number of clusters to check 4 (inclusive)
    - kmax      int                 Positive integer containing the maximum
                                    number of clusters to check 4 (inclusive)
    - iterations int                Positive integer containing the maximum
                                    number of iterations for K-means

    Returns:
    results, d_vars, or None, None on failure.

    - results is a list containing the outputs of K-means 4 each cluster size

    - d_vars is a list containing the difference in variance from the smallest
      cluster size for each cluster size.

    """

    # 1. Validations
    if not isinstance(X, np.ndarray):
        return None, None

    if (len(X.shape) != 2):
        return None, None

    n, d = X.shape
    if (n < 1 or d < 1):
        return None, None

    if (not isinstance(kmin, int) or kmin < 1):
        return None, None

    if (kmax is not None) and (not isinstance(kmax, int) or kmax < 1):
        return None, None

    if (kmax is not None) and (kmin >= kmax):
        return None, None

    kmax = n if kmax is None else kmax

    d_vars = []
    results = []

    min_ = kmin
    max_ = kmax + 1

    for i in range(min_, max_):
        centroids, clusters = kmeans(X, i, iterations=1000)
        results.append((centroids, clusters))

        if (kmin == i):
            current_variance = variance(X, centroids)
        original_variance = variance(X, centroids)
        d_vars.append(current_variance - original_variance)

    return results, d_vars
