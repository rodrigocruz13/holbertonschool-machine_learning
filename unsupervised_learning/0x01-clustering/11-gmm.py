#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import sklearn.mixture as sklmix


def gmm(X, k):
    """
    Function that calculates a GMM from a dataset

    Args:
    - X             numpy.ndarray       Array of shape (n, d) containing the
                                        dataset
        - n         int                 Number of data points
        - d         int                 Number of dim for each data point

    - k             int                 Positive integer containing the number
                                        of clusters

    Returns:
    pi, m, S, clss, bic
    - pi            numpy.ndarray       Array of shape (k,) containing the
                                        cluster priors
    - m             numpy.ndarray       Array of shape (k, d) containing the
                                        centroid means
    - S             numpy.ndarray       Array of shape (k, d, d) containing
                                        the covariance matrices
    - clss          numpy.ndarray       Array of shape (n,) containing the
                                        cluster indices for each data point
    - bic           numpy.ndarray       Array of shape (kmax - kmin + 1)
                                        containing the BIC value for each
                                        cluster size tested
    """

    gmm_ = sklmix.GaussianMixture(n_components=k).fit(X)

    priors = gmm_.weights_
    centroids = gmm_.means_
    covariances = gmm_.covariances_
    clusters_labels = gmm_.predict(X)
    bic_value = gmm_.bic(X)

    return priors, centroids, covariances, clusters_labels, bic_value
