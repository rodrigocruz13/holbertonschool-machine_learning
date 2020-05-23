#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import sklearn.cluster as skl


def kmeans(X, k):
    """
    Function that performs K-means on a dataset:

    Args:
    - X             numpy.ndarray       Array of shape (n, d) containing the
                                        dataset
        - n         int                 Number of data points
        - d         int                 Number of dim for each data point

    - k             int                 Positive integer containing the number
                                        of clusters

    Returns:
    C, clss,   or    None, None on failure

    - C             np.ndarray          Array of shape (k, d) containing the
                                        centroid means for each cluster
    - clss          np.ndarray          Array of shape (n,) containing the
                                        index of the cluster in C that each
                                        data point belongs to
    """

    # Validations

    k = skl.KMeans(n_clusters=k).fit(X)
    centers = k.cluster_centers_
    clusters_classes = k.labels_

    return centers, clusters_classes
