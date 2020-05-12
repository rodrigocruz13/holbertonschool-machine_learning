#!/usr/bin/env python3
""" module """

import numpy as np


def mean_cov(X):
    """
    Function that calculates the mean and covariance of a data set:

    Args:
        - X:        numpy.ndarray   Array of shape (n,d) containing
                    the data set:
            - n     int             number of data points
            - d     int             number of dimensions in each data point

        If X is not a 2D numpy.ndarray, raise a TypeError with the message:
        X must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with the message
        X must contain multiple data points

    Returns:
        - mean, cov:
            - mean: numpy.ndarray   Array of shape (1, d) containing the
                                    mean of the data set
            - cov:  numpy.ndarray   Array of shape (d, d) containing the
                                    covariance matrix of the data set
    """
    if(isinstance(X, type(None))):
        raise TypeError('X must be a 2D numpy.ndarray')

    if (not isinstance(X, np.ndarray)) or (len(X.shape) != 2):
        raise TypeError('X must be a 2D numpy.ndarray')

    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    mean = X.mean(axis=0)
    mean = np.reshape(mean, (-1, X.shape[1]))

    n = X.shape[0] - 1
    x = X - mean
    cov = np.dot(x.T, x) / n

    return mean, cov
