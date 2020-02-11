#!/usr/bin/env python3
"""
Class for the poisson distribution
"""


class Poisson:
    """ Class """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Poisson
        Args:
            data (lst): List of the data to be used to estimate the dist
            lambtha (int): Expected number of events in a given time frame
        Returns:
        """
        self.lambtha = lambtha
        π = 3.1415926536
        e = 2.7182818285

        λ = float(lambtha)

        if (data is None or (type(data) is list and len(data) == 0)):
            if λ < 0:
                raise ValueError("lambtha must be a positive value")

        else:
            if λ < 0:
                raise ValueError("lambtha must be a positive value")
            if type(data) is not list:
                raise ValueError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            λ = float(sum(data) / len(data))
            self.lambtha = λ

    def erf(x):
        """
        is the "error function" encountered in integrating the normal distr
        (which is a normalized form of the Gaussian function). Erf can also
        be defined as a Maclaurin series.
        Args:
            x (float): Value
        Returns:
               (float): An error aproximation using the Maclaurin series.
        """

        π = 3.1415926536
        root_π = π**(1/2)
        x3 = x**3
        x5 = x**5
        x7 = x**7
        x9 = x**9

        return (2/root_π) * x - x3/3 + x5/10 - x7/42 + x9/216
