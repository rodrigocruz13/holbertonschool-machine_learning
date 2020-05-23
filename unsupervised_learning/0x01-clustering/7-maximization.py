#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np

pdf = __import__('5-pdf').pdf


def maximization(X, g):
    """
    Function that calculates the maximization step in the EM algorithm 4 a GMM

    Args:
    - X     numpy.ndarray       Array of shape (n, d) containing the data
    - g     numpy.ndarray       Array of shape (k, n) containing the posterior
                                probabilities for each data point in each
                                cluster
    Returns:
    pi, m, S, or None, None, None on failure

    - pi    numpy.ndarray       Array of shape (k,) containing the updated
                                priors for each cluster
    - m     numpy.ndarray       Array of shape (k, d) containing the updated
                                centroid means for each cluster
    - S     numpy.ndarray       Array of shape (k, d, d) containing the
                                updated covariance matrices for each cluster
    """

    try:
        if (not isinstance(X, np.ndarray)):
            return None, None, None

        if (not isinstance(g, np.ndarray)):
            return None, None, None

        if not (np.isclose(np.sum(g, axis=0), 1).all):
            return None, None, None

        n, d = X.shape
        k = g.shape[0]
        if (g.shape != (k, n)) or (X.shape != (n, d)):
            return None, None, None

        if (n < 1) or (d < 1) or (k < 1) or (k > n):
            return None, None, None

        pi = np.zeros((k,))
        m = np.zeros((k, d))
        S = np.zeros((k, d, d))

        for i in range(k):

            # 1. Calculate priors
            g_i_acum = np.sum(g[i], axis=0)
            pi[i] = g_i_acum / n

            # 2 Calculate centroids
            gi_alt = g[i].reshape(1, n)
            m[i] = np.matmul(gi_alt, X).sum(axis=0) / g_i_acum

            # 3. Calculate covariances (matrix of shape k, d d)
            diff = (X - m[i])
            S[i] = np.dot(g[i] * diff.T, diff) / g_i_acum

        return pi, m, S

    except BaseException:
        return None, None, None
