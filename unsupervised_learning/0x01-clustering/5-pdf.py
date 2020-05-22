#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np


def pdf(X, m, S):
    """
    Funtion that calculates the probability density function of a
    Gaussian distribution:

    Args:
    - X     numpy.ndarray       Array of shape (n, d) containing the data
                                points whose PDF should be evaluated
    - m     numpy.ndarray       Array of shape (d,) containing the mean of the
                                distribution
    - S     numpy.ndarray       Array of shape (d, d) containing the covariance
                                of the distribution
    Returns:
    P, or None on failure
    - P     numpy.ndarray       Array of shape (n,) containing the PDF values
                                for each data point
    All values in P should have a minimum value of 1e-300
    """

    try:
        if (not isinstance(X, np.ndarray)):
            return None

        if (not isinstance(m, np.ndarray)):
            return None

        if (not isinstance(S, np.ndarray)):
            return None

        if (X.ndim != 2) or (m.ndim != 1) or (S.ndim != 2):
            return None

        d = X.shape[1]
        if (m.shape != (d,)) or (S.shape != (d, d)):
            return None

        n = X.shape[0]
        if (n < 1) or (d < 1):
            return None

        # https://bit.ly/3cXVLPN

        inverse = np.linalg.inv(S)
        determinant = np.linalg.det(S)

        sqr = np.sqrt(pow(2 * np.pi, d) * determinant)
        fac = np.einsum('...k,kl,...l->...', (X - m), inverse, (X - m))
        exp = np.exp((-1 / 2) * fac)
        pdf = (1 / sqr) * exp

        pdf = np.maximum(pdf, 1e-300)
        return pdf

    except BaseException:
        return None
