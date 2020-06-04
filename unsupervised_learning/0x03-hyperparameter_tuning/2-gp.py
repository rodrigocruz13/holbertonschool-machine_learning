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
        The covariance
        kernel    : numpy.ndarray
                    Array shape (m, n)
        """

        # https://krasserm.github.io/2018/03/19/gaussian-processes/

        sqr_sumx1 = np.sum(X1**2, 1).reshape(-1, 1)
        sqr_sumx2 = np.sum(X2**2, 1)
        sqr_dist = sqr_sumx1 + sqr_sumx2 - 2 * np.dot(X1, X2.T)
        kernel = self.sigma_f**2 * np.exp(-0.5 / self.l**2 * sqr_dist)
        return kernel

    def predict(self, X_s):
        """
        Class method.
        Predicts the mean and std deviation of points in a Gaussian process

        Arguments
        ---------
        - X_s     : numpy.ndarray
                    Array of shape (s, 1) containing all of the points whose
                    mean and standard deviation should be calculated
                s : int
                    is the number of sample points

        Returns
        -------
        mu, sigma
        mu        : numpy.ndarray
                    array of shape (s,) containing the mean for each point in
                    X_s, respectively
        sigma     : numpy.ndarray
                    Array of shape (s,) containing the standard deviation for
                    each point in X_s, respectively
        """

        # https://gist.github.com/stober/4964727
        # https://krasserm.github.io/2018/03/19/gaussian-processes/

        s = X_s.shape[0]
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s) + np.ones(s) - np.eye(s)
        K_inv = np.linalg.inv(K)

        # Equation (4)
        μ = (K_s.T.dot(K_inv).dot(self.Y)).flatten()

        # Equation (5)
        cov_s = (K_ss - K_s.T.dot(K_inv).dot(K_s))
        cov_s = np.diag(cov_s)

        return μ, cov_s

    def update(self, X_new, Y_new):
        """
        Class method.
        Updates a Gaussian Process
        Updates the public instance attributes X, Y, and K

        Arguments
        ---------
        - X_new   : numpy.ndarray
                    array of shape (1,) that represents the new sample point
        - Y_new   : numpy.ndarray
                    array of shape (1,) that represents the new sample function
                    value

        Returns
        -------
        The variables X, Y, K updated
        """

        # print(X_new)
        # print(self.X)

        self.X = np.append(self.X, X_new)
        # self.Y = self.Y.T
        self.X = np.reshape(self.X, (-1, 1))

        self.Y = np.append(self.Y, Y_new)
        # self.Y = self.Y.T
        self.Y = np.reshape(self.Y, (-1, 1))
        self.K = self.kernel(self.X.reshape(-1, 1), self.X.reshape(-1, 1))
