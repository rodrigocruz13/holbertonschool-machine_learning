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

        if(isinstance(data, type(None))):
            raise TypeError('data must be a 2D numpy.ndarray')

        if (not isinstance(data, np.ndarray)) or (len(data.shape) != 2):
            raise TypeError('data must be a 2D numpy.ndarray')

        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")

        dataT = data.T

        mean = dataT.mean(axis=0)
        mean = np.reshape(mean, (-1, dataT.shape[1]))

        n = dataT.shape[0] - 1
        x = dataT - mean
        cov = np.dot(x.T, x) / n

        self.mean = mean
        self.cov = cov
