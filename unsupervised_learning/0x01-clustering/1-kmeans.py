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
    try:

        if not isinstance(X, np.ndarray) or not isinstance(k, int):
            return None

        if (k < 1):
            return None

        # generate cluster centers (CC)
        d = X.shape[1]
        min_ = np.amin(X, axis=0)
        max_ = np.amax(X, axis=0)
        CC = np.random.uniform(low=min_, high=max_, size=(k, d))

    except BaseException:
        return None

    return CC


def kmeans(X, k, iterations=1000):
    """
    Function that performs K-means on a dataset:

    Args:
    - X             numpy.ndarray       Array of shape (n, d) containing the
                                        dataset
        - n         int                 Number of data points
        - d         int                 Number of dim for each data point

    - k             int                 Positive integer containing the number
                                        of clusters
    - iterations    int                 Positive integer containing the max
                                        number of iterations that should be
                                        performed

    Initialize the cluster centroids using a multivariate uniform distribution
    (based on 0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
    its centroid
    You should use numpy.random.uniform exactly twice
    You may use at most 2 loops

    Returns:
    C, clss,   or    None, None on failure

    - C             np.ndarray          Array of shape (k, d) containing the
                                        centroid means for each cluster
    - clss          np.ndarray          Array of shape (n,) containing the
                                        index of the cluster in C that each
                                        data point belongs to
    If no change occurs between iterations, your function should return

    """

    # Validations

    try:
        if (not isinstance(X, np.ndarray)) or (len(X.shape) != 2):
            return None, None

        if not isinstance(iterations, int) or (iterations < 1):
            return None, None

        n, d = X.shape
        if not isinstance(k, int) or (k < 1) or (n < k):
            return None, None

        # Generate centers for each cluster
        m = np.amin(X, axis=0)
        M = np.amax(X, axis=0)

        old_centers = np.random.uniform(m, M, size=(k, d))
        new_centers = np.ndarray.copy(old_centers)
        if (old_centers.any() is None):
            return None, None

        error = 100

        # distances = np.zeros((n, k))
        for iter_ in range(iterations):
            # 1. Generate distances
            distances = np.sqrt(
                ((X - new_centers[:, np.newaxis])**2).sum(axis=-1))

            # 2. assign points to clusters
            clusters = np.argmin(distances, axis=0)

            # 3. calculate new centroids
            for j in range(k):
                # If a cluster has no data points, reinitialize its centroid
                if (X[clusters == j].size == 0):
                    new_centers[j, :] = np.random.uniform(m, M, size=(1, d))
                    if (new_centers.any() is None):
                        return None, None
                else:
                    new_centers[j, :] = (X[clusters == j].mean(axis=0))

            new_error = np.linalg.norm(new_centers - old_centers)
            C = new_centers
            clss = clusters

            # If no change occurs between iterations, your function
            # should return

            if ((error - new_error == 0) or (iter_ == iterations)):
                return C, clss
            error = new_error

        return C, clss

    except BaseException as e:
        print(e)
        return None, None
