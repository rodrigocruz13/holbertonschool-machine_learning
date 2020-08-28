#!/usr/bin/env python3

"""
class Bayesian Optimization
"""

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization():
    """
        Performs Bayesian optimization on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """
        Class constructor

        Arguments
        ---------
        - f          :
                       is the black-box function to be optimized
        - X_init:    : numpy.ndarray
                       Array of shape (t, 1) representing the inputs already
                       sampled with the black-box function
        - Y_init     : numpy.ndarray
                       Array of shape (t, 1) representing the outputs of the
                       black-box function for each input in X_init
                t    : int
                       The number of initial samples
        - bbounds    : tuple
                       (min, max) representing the bounds of the space in
                       which to look for the optimal point
        - ac_samples : int
                       number of samples that should be analyzed during
                       acquisition
        - l          : int
                       length parameter for the kernel
        - sigma_f    : int float
                       standard deviation given to the output of the black-box
                       function
        - xsi        : float
                       exploration-exploitation factor for acquisition
        - minimize   : bool
                       Says whether optimization should be performed for
                       minimization (True) or maximization (False)

        Returns
        -------
        Sets the following public instance attributes:
        - f          : the black-box function
        - gp         : an instance of the class GaussianProcess
        - X_s        : numpy.ndarray of shape (ac_samples, 1) containing all
                       acquisition sample points, evenly spaced between min
                       and max
        - xsi        : the exploration-exploitation factor
        - minimize   : a bool for minimization versus maximization
        """

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.zeros((ac_samples, 1))
        self.X_s = np.linspace(start=bounds[0],
                               stop=bounds[1],
                               num=ac_samples,
                               endpoint=True)
        self.X_s = self.X_s.reshape(ac_samples, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location:
        (Uses the Expected Improvement acquisition function)

        Arguments
        ---------

        Returns
        --------
        X_next, EI
        - X_next    : numpy.ndarray
                      Array of shape (1,) representing the next best sample
                      point
        - EI        : numpy.ndarray
                      Array of shape (ac_samples,) containing the expected
                      improvement of each potential sample
        """

        # https://krasserm.github.io/2018/03/21/bayesian-optimization/
        # def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):

        m_sample, sigma = self.gp.predict(self.X_s)

        # Needed for noise-based model,
        # otherwise use np.max(Y_sample).
        # See also section 2.4 in [...]

        sam = np.min(self.gp.Y) if self.minimize else np.max(self.gp.Y)
        imp = (sam - m_sample if self.minimize else m_sample - sam) - self.xsi

        with np.errstate(divide='warn'):
            Z = imp / sigma
            EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
            EI[sigma == 0.0] = 0.0

        X_next = self.X_s[np.argmax(EI)]
        return X_next, EI

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function:

        Arguments
        ---------

        - iterations  : int
                        is the maximum number of iterations to perform
                        If the next proposed point is one that has already
                        been sampled, optimization should be stopped early

        Returns
        -------
        X_opt, Y_opt
        - X_opt       : numpy.ndarray
                        Array of shape (1,) representing the optimal point
        - Y_opt       : numpy.ndarray
                        Array of shape (1,) representing the optimal function
                        value
        """

        # https://krasserm.github.io/2018/03/21/bayesian-optimization/

        X_opt = 0
        Y_opt = 0

        for _ in range(iterations):

            # Find the next best sample
            X_next = self.acquisition()[0]

            # if X_next already sampled in gp.X, ommit
            if (X_next in self.gp.X):
                break

            else:
                # get Y_next, evaluate X_next is black box function
                Y_next = self.f(X_next)

                # updates a GP
                self.gp.update(X_next, Y_next)

                # if minimizing save the least otherwise save the largest
                if (Y_next < Y_opt) and (self.minimize):
                    X_opt, Y_opt = X_next, Y_next

                if not self.minimize and Y_next > Y_opt:
                    X_opt, Y_opt = X_next, Y_next

        # removing last element
        self.gp.X = self.gp.X[:-1]

        return X_opt, Y_opt
