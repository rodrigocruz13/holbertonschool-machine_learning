#!/usr/bin/env python3

"""
class Gaussian Process
"""

import numpy as np


class GaussianProcess():
    """
        Represents a noiseless 1D Gaussian process
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor.
        Sets the public instance attributes X, Y, l, and sigma_f corresponding
        to the respective constructor inputs
        Sets the public instance attribute K, representing the current
        covariance kernel matrix for the Gaussian process

        Arguments
        ---------
        - X_init  : numpy.ndarray
                    Array of shape (t, 1) representing the inputs already
                    sampled with the black-box function
        - Y_init  : numpy.ndarray
                    Array of shape (t, 1) representing the outputs of the
                    black-box function for each input in X_init
                t : int
                    t is the number of initial samples
        - l       : float
                    length parameter for the kernel
        - sigma_f : int - float
                    is the standard deviation given to the output of the
                    black-box function

        Returns
        -------
        The variables X, Y, l, and sigma_f initializated
        """

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f

        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Class method.
        Calculates the covariance kernel matrix between two matrices.
        The kernel should use the Radial Basis Function (RBF)

        Arguments
        ---------
        - X1      : numpy.ndarray
                    Array of shape (m, 1)
        - X2      : numpy.ndarray
                    Array of shape (n, 1)

        Returns
        -------
        The covariance matrix
        kernel    : numpy.ndarray
                    Array shape (m, n)
        """

        # https://krasserm.github.io/2018/03/19/gaussian-processes/
        # K(xᵢ, xⱼ) = σ² exp((-0.5 / 2l²)(xᵢ − xⱼ)ᵀ (xᵢ − xⱼ))

        σ2 = self.sigma_f ** 2
        l2 = self.l ** 2

        sqr_sumx1 = np.sum(X1 ** 2, 1).reshape(-1, 1)
        sqr_sumx2 = np.sum(X2 ** 2, 1)
        sqr_dist = sqr_sumx1 - 2 * np.dot(X1, X2.T) + sqr_sumx2

        kernel = σ2 * np.exp(-0.5 / l2 * sqr_dist)
        return kernel
