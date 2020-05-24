#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Function that performs the expectation maximization for a GMM:

    Args:
    - X             numpy.ndarray       Array of shape (n, d) with the data
    - k             int                 positive int with the # of clusters
    - iterations    int                 positive int with the maximum number
                                        of iterations for the algorithm
    - tol           float               non-negative float with the tolerance
                                        of the log likelihood, used to
                                        determine early stopping i.e. if the
                                        difference is less than or equal to
                                        tol you should stop the algorithm
    - verbose       bool                boolean that determines if you should
                                        print information about the algorithm

                    If True, print Log Likelihood after {i} iterations: {l}
                    every 10 iterations and after the last iteration
                        {i} is the number of iterations of the EM algorithm
                        {l} is the log likelihood
                    You should use:
                    initialize = __import__('4-initialize').initialize
                    expectation = __import__('6-expectation').expectation
                    maximization = __import__('7-maximization').maximization

Returns: pi, m, S, g, l, or None, None, None, None, None on failure
    - pi            numpy.ndarray       Array of shape (k,) containing the
                                        priors for each cluster
    - m             numpy.ndarray       Array of shape (k, d) containing the
                                        centroid means for each cluster
    - S             numpy.ndarray       Array of shape (k, d, d) containing
                                        the cov matrices for each cluster
    - g             numpy.ndarray       Array of shape (k, n) containing the
                                        probabilities for each data point in
                                        each cluster
    - l                                 is the log likelihood of the model
    """

    try:
        if (not isinstance(X, np.ndarray)):
            return None, None, None, None, None

        if (not isinstance(k, int)):
            return None, None, None, None, None

        if (not isinstance(iterations, int)):
            return None, None, None, None, None

        if (not isinstance(tol, float)):
            return None, None, None, None, None

        if (not isinstance(verbose, bool)):
            return None, None, None, None, None

        if (X.ndim != 2):
            return None, None, None, None, None

        n, d = X.shape
        if (n < 1) or (d < 1) or (k < 1) or (k > n) or (iterations < 1):
            return None, None, None, None, None

        pi = np.zeros((k,))
        m = np.zeros((k, d))
        S = np.zeros((k, d, d))

        return pi, m, S, 1, 1

    except BaseException:
        return None, None, None, None, None