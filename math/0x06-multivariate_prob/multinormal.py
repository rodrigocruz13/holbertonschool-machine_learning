#!/usr/bin/env python3
"""
Class for the Multinirmal probabilities
"""

import numpy as np


class MultiNormal:
    """
    Class that represents a Multivariate Normal distribution:
    """

    def __init__(self, data):
        """
        class constructor

        Args:
            - Data:     numpy.ndarray       Array of shape (d, n) containing
                                            the dataset:
                - n     int             number of data points
                - d     int             number of dimensions in each data point

            If Data is not a 2D numpy.ndarray, raise a TypeError with the msg:
            data must be a 2D numpy.ndarray
            If n is less than 2, raise a ValueError with the message
            data must contain multiple data points

            Set the public instance variables:
            mean - a numpy.ndarray of shape (d, 1) containing the mean of data
            cov - a numpy.ndarray of shape (d, d) containing the covariance 
            matrix data
        """

        """if(isinstance(data, type(None))):
            raise TypeError('data must be a 2D numpy.ndarray')
        """

        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        if data.shape[1] < 2:
            raise TypeError("data must contain multiple data points")

        # Set the public instance variables

        d = data.shape[0]
        n = data.shape[1] - 1

        mean = np.mean(data.T, axis=0).reshape(1, d)
        x = data.T - mean
        cov = np.dot(x.T, x) / n

        self.d = d
        self.mean = mean.T
        self.cov = cov.T

