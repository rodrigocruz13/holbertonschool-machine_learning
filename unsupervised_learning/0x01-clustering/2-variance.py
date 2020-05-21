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

    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None

    if (k < 1):
        return None

    # generate cluster centers (CC)
    try:
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

    if not isinstance(X, np.ndarray) or not isinstance(k, int):
        return None, None

    if (k < 1) or (iterations < 1):
        return None, None

    # Generate centers for each cluster
    old_centers = initialize(X, k)

    if (old_centers.any() is None):
        return None, None

    try:

        new_centers = np.ndarray.copy(old_centers)
        iter_ = 0
        error = 100
        while iter_ < iterations:

            # 1. Generate distances
            deltas = X[:, np.newaxis, :] - new_centers
            distances = np.sqrt(np.sum((deltas) ** 2, 2))

            # 2. assign points to clusters
            clusters = distances.argmin(1)

            # 3. calculate new centroids
            for j in range(k):
                # If a cluster has no data points, reinitialize its centroid
                if (X[clusters == j].size == 0):
                    d = X.shape[1]
                    min_ = np.amin(X, axis=0)
                    max_ = np.amax(X, axis=0)
                    new_centers[j, :] = np.random.uniform(min_,
                                                          max_,
                                                          size=(1, d))
                else:
                    new_centers[j, :] = (X[clusters == j].mean(axis=0))

            new_error = np.linalg.norm(new_centers - old_centers)
            C = new_centers
            clss = clusters
            iter_ = iter_ + 1

            # If no change occurs between iterations, your function
            # should return

            if (error - new_error == 0):
                return C, clss
            error = new_error

        return C, clss

    except BaseException:
        return None, None


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

    try:
        # https://paris-swc.github.io/advanced-numpy-lesson/05-kmeans.html

        deltas = X[:, np.newaxis] - C
        dist = np.sqrt(np.sum((deltas) ** 2, axis=2))
        min_dist = np.min(dist, axis=1)
        var = np.sum(min_dist ** 2)
        return var

    except BaseException:
        return None
