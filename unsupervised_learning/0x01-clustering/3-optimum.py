#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Funtion that tests for the optimum number of clusters by variance:

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
    results, d_vars, or None, None on failure

    - results is a list containing the outputs of K-means 4 each cluster size

    - d_vars is a list containing the difference in variance from the smallest
      cluster size for each cluster size

    """
    try:
        # 1. Validations

        if not isinstance(X, np.ndarray):
            return None, None

        if (len(X.shape) != 2):
            return None, None

        if (X.shape[0] < 1 or X.shape[1] < 1):
            return None, None

        if (not isinstance(kmin, int)) or (not isinstance(kmax, int)):
            return None, None

        if (kmin < 1) or (kmax < 1):
            return None, None

        if (kmin >= kmax) or (kmax >= X.shape[0]):
            return None, None

        if (not isinstance(iterations, int)) or iterations < 1:
            return None, None

        d_vars = []
        results = []

        centroids, clusters = kmeans(X, kmin, iterations)
        original_variance = variance(X, centroids)

        min_ = kmin
        max_ = kmax + 1

        for i in range(min_, max_):
            centroids, clusters = kmeans(X, i, iterations)
            results.append((centroids, clusters))

            current_variance = variance(X, centroids)
            d_vars.append(original_variance - current_variance)

        return results, d_vars

    except BaseException:
        return None, None
