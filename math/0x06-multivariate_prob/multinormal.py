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

    def pdf(self, x):
        """
        Method that calculates the PDF at a data point:
        Args:
            - x:    np.ndarray  Array of shape (d, 1) with the data point
                                to calculate PDF
                    d           Int. number of dimensions of the Multinomial
                                instance
            If x is not a numpy.ndarray, raise a TypeError with the message
            x must by a numpy.ndarray
            If x is not of shape (d, 1), raise a ValueError with the message
            x mush have the shape ({d}, 1)

        Returns: The value of the PDF
        """

        d = self.cov.shape[0]

        if(isinstance(x, type(None))):
            raise TypeError('x must be a numpy.ndarray')

        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        if (x.shape[0] != d):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        if (len(x.shape) != 2) or (x.shape[1] != 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # source:
        # https://peterroelants.github.io/posts/multivariate-normal-primer/
        # p(x∣μ,Σ)=[1√(2π)^d|Σ|] * exp((−1/2)(x−μ)^T(Σ^−1(x−μ))

        mean = self.mean
        cov = self.cov

        x_m = x - mean
        pdf = (1. / (np.sqrt((2 * np.pi)**d * np.linalg.det(cov)))
               * np.exp(-(np.linalg.solve(cov, x_m).T.dot(x_m)) / 2))

        return pdf[0][0]
