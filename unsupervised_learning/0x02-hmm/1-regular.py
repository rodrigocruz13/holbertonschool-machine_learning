#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np


def markov_chain(P, s, t=1):
    """
    Function that determines the probability of a markov chain being in a
    particular state after a specified number of iterations:

    Args:
    - P     np.ndarray          It is a 2d array of shape (n, n) representing
                                the transition matrix.
                                P[i, j] is the probability of transitioning
                                from state i to state j
        - n     int             Number of states in the markov chain
    - s     np.ndarray          Array of shape (1, n) representing the prob
                                of starting in each state
    - t     int                 Number of iterations that the markov chain has
                                been through
    Returns:
                                a numpy.ndarray of shape (1, n) representing
                                the probability of being in a specific state
                                after t iterations, or None on failure
    """

    try:

        if (not isinstance(P, np.ndarray)) or (not isinstance(s, np.ndarray)):
            return None

        if (not isinstance(t, int)):
            return None

        if (P.ndim != 2) or (s.ndim != 2) or (t < 1):
            return None

        n = P.shape[0]
        if (P.shape != (n, n)) or (s.shape != (1, n)):
            return None

        while (t > 0):
            s = np.matmul(s, P)
            t = t - 1
        return s
    except BaseException:
        return None, None


def regular(P):
    """
    Function that determines the steady state probabilities of a regular
    markov chain:

    Args:
    - P         numpy.ndarray       A square 2D array of shape (n, n)
                                    representing the transition matrix
                                    P[i, j] is the probability of
                                    transitioning from state i to state j
    - n         int                 Number of states in the markov chain
    Returns:
    - regular   numpy.ndarray       Array of shape (1, n) containing the
                                    steady state probabilities, or None on
                                    failure
    """

    np.warnings.filterwarnings('ignore')
    # Avoid this warning: Line 92.  np.linalg.lstsq(a, b)[0]

    try:

        if (not isinstance(P, np.ndarray)):
            return None

        if (P.ndim != 2):
            return None

        n = P.shape[0]
        if (P.shape != (n, n)):
            return None

        if ((np.sum(P) / n) != 1):
            return None

        if ((P > 0).all()):  # Are all elements of P positive ?
            a = np.eye(n) - P
            a = np.vstack((a.T, np.ones(n)))
            b = np.matrix([0] * n + [1]).T
            regular = np.linalg.lstsq(a, b)[0]
            return regular.T

        return None

    except BaseException:
        return None
