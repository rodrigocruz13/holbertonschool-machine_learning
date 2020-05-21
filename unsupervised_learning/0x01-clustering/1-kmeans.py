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


def asign_clusters(X, distances, k, new_centers):
    """
    Funtion that assings to a data point the number of the closer cluster
    according to the distance calculated

    Args
    - X             numpy.ndarray       Array of shape (n, d) containing the
                                        dataset
        - n         int                 Number of data points
        - d         int                 Number of dim for each data point

    - k             int                 Positive integer containing the number
                                        of clusters
    - distances     numpy.ndarray       Array of shape (n, k) with the distance
                                        of each point to each cluster

    Returns
    - clusters     list                 list of shape (n, ) witht the assigned
                                        cluster number for each datapoint
    """

    clusters = np.argmin(distances, axis=1)
    used_clusters = np.unique(clusters, return_counts=True)[0]
    clusters_with_no_data = np.setdiff1d(np.array(range(k)), used_clusters)

    # If a cluster has no data points, reinitialize its centroid
    while (len(clusters_with_no_data) > 0):

        for i in clusters_with_no_data:
            temp_centers = initialize(X, k)
            new_centers[i] = temp_centers[(i - 1) % k]
            # new_centers[i] = temp_centers[i]

        # asign each datapoint to a clusters
        deltas = X[:, np.newaxis] - new_centers
        distances = np.sqrt(np.sum((deltas) ** 2, axis=2))

        clusters = np.argmin(distances, axis=1)
        used_clusters = np.unique(clusters, return_counts=True)[0]
        clusters_with_no_data = np.setdiff1d(np.array(range(k)), used_clusters)
    return clusters


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

        error = 1
        iter_ = 0
        while (error != 0 and iter_ < iterations):

            # 1. Measure the distance to every center
            deltas = X[:, np.newaxis] - new_centers
            distances = np.sqrt(np.sum((deltas) ** 2, axis=2))

            # 2. Assign points to each cluster
            clust = np.argmin(distances, axis=1)

            # 2a. If a cluster has no data points, reinitialize its centroid
            used_clusts = np.unique(clust, return_counts=True)[0]
            clust_with_0_data = np.setdiff1d(np.array(range(k)), used_clusts)

            if (len(clust_with_0_data) > 0):
                temp_clusters = initialize(X, k)
                if (temp_clusters is None):
                    return None, None

                for i in clust_with_0_data:
                    new_centers[i] = temp_clusters[(i - 1) % k]

            # 2b.  Asign data to each cluster
            deltas = X[:, np.newaxis] - new_centers
            distances = np.sqrt(np.sum((deltas) ** 2, axis=2))
            clust = np.argmin(distances, axis=1)

            # 3. Calculate & update new center (mean) for all clusters
            old_centers = np.ndarray.copy(new_centers)
            for i in range(k):
                new_centers[i] = np.mean(X[clust == i], axis=0)

            error = np.linalg.norm(new_centers - old_centers)
            iter_ += 1

        C = new_centers
        clss = clust

        return C, clss

    except BaseException:
        return None, None