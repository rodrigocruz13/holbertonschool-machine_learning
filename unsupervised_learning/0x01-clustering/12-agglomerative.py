#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Function that performs agglomerative clustering on a dataset with Ward
    linkage. Displays the dendrogram with each cluster displayed in a different
    color

    Args:
    - X             numpy.ndarray       Array of shape (n, d) containing the
                                        dataset
        - n         int                 Number of data points
        - d         int                 Number of dim for each data point

    - dist          int                 is the maximum cophenetic distance for
                                        all clusters

    Returns:
    - clss          numpy.ndarray       a numpy.ndarray of shape (n,)
                                        containing the cluster indices for
                                        each data point
    """

    linkage = scipy.cluster.hierarchy.linkage
    dendogram = scipy.cluster.hierarchy.dendrogram
    cluster = scipy.cluster.hierarchy.fcluster

    # 1 generate the linkage matrix
    Z = linkage(X, "ward")

    # 2. Generate dendogram
    dendogram(Z, color_threshold=dist, no_labels=False, ax=None)
    plt.show()

    # 3. Calculate clusters
    clss = cluster(Z, dist, criterion="distance")
    return clss
