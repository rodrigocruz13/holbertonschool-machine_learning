#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Funtion that calculates the probability density function of a
    Gaussian distribution:

    Args:
    - X     numpy.ndarray       Array of shape (n, d) containing the data
    - pi    numpy.ndarray       Array of shape (k,) containing the priors for
                                each cluster
    - m     numpy.ndarray       Array of shape (k, d) containing the mean of
                                the distribution
    - S     numpy.ndarray       Array of shape (k, d, d) containing the
                                covariance of the distribution
    Returns:
    g, l, or None, None on failure

    - g     numpy.ndarray       Array of shape (k, n) containing the posterior
                                probabilities for each data point in each
                                cluster
    - l     float               total log likelihood
    """

    try:
        if (not isinstance(X, np.ndarray)):
            return None, None

        if (not isinstance(m, np.ndarray)):
            return None, None

        if (not isinstance(S, np.ndarray)):
            return None, None

        if (X.ndim != 2) or (m.ndim != 2) or (S.ndim != 3):
            return None, None

        k = pi.shape[0]
        if (pi.shape != (k,)):
            return None, None

        if (pi.sum() != 1):
            return None, None

        n, d = X.shape
        if (m.shape != (k, d)) or (S.shape != (k, d, d)):
            return None, None

        if (n < 1) or (d < 1):
            return None, None

        # https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html

        g = np.zeros((k, n))
        for i in range(k):
            if (pdf(X, m[i], S[i]) is None):
                return None, None
            g[i] = pdf(X, m[i], S[i]) * pi[i]

        log_likelihood = np.sum(np.log(np.sum(g, axis=0)))
        posterior_probabilities = g / np.sum(g, axis=0)

        return posterior_probabilities, log_likelihood

    except BaseException:
        return None, None
