#!/usr/bin/env python3

"""
Class for the Multinirmal probabilities
"""

import numpy as np


class MultiNormal:
    """ Represents a Multivariate Normal distribution"""

    def __init__(self, data):
        """
        class constructor

        Args:
            - data:     numpy.ndarray       Array of shape (d, n) containing
                                            the dataset:
                - n     int             number of data points
                - d     int             number of dimensions in each data point

            If data is not a 2D numpy.ndarray, raise a TypeError with the msg:
            data must be a 2D numpy.ndarray
            If n is less than 2, raise a ValueError with the message
            data must contain multiple data points

            Set the public instance variables:
            mean - a numpy.ndarray of shape (d, 1) containing the mean of data
            cov -  a numpy.ndarray of shape (d, d) containing the covariance
            matrix data
        """

        if(isinstance(data, type(None))):
            raise TypeError('data must be a 2D numpy.ndarray')

        if (not isinstance(data, np.ndarray)) or (len(data.shape)) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if (data.shape[1] < 2):
            raise ValueError("data must contain multiple data points")

        data = data.T
        mean = data.mean(axis=0)
        mean = np.reshape(mean, (-1, data.shape[1]))

        n = data.shape[0] - 1
        x = data - mean
        cov = np.dot(x.T, x) / n

        self.mean = mean.T
        self.cov = cov
