#!/usr/bin/env python3
"""
Class for the Exponential distribution
"""


class Exponential:
    """ Class """

    def __init__(self, data=None, lambtha=1.):
        """
        Initialize Exponential
        Args:
            - data (lst): List of the data to be used to estimate the dist
            - lambtha (int): Expected number of events in a given time frame
        Returns:
            Nothing. Just initialize the variables
        """
        self.lambtha = float(lambtha)
        self.π = 3.1415926536
        self.e = 2.7182818285

        λ = self.lambtha

        if data is None:
            if λ <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")

            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            λ = float(1 / (sum(data) / len(data)))
            self.lambtha = λ

    def erf(self, x):
        """
        is the "error function" encountered in integrating the normal distr
        (which is a normalized form of the Gaussian function). Erf can also
        be defined as a Maclaurin series.
        Args:
            x (float): Value
        Returns:
               (float): An error aproximation using the Maclaurin series.
        """

        π = self.π
        root_π = π**(1 / 2)
        x3 = x**3
        x5 = x**5
        x7 = x**7
        x9 = x**9

        return (2 / root_π) * x - x3 / 3 + x5 / 10 - x7 / 42 + x9 / 216

    def factorial(self, k):
        """
        Returns the factorial of a number.
        Args:
            k (int): Value
        Returns:
            num! (int): An error aproximation using the Maclaurin series.
        """
        n = int(k)
        fact = 1

        for n in range(1, n + 1):
            fact = fact * n
        return fact

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given number of “successes”
        Args:
            x (float):  is the time period

        Returns:
               pdf (float): The PDF value for x.
               0 if x is out of range
        """

        k = x
        λ = self.lambtha
        e = self.e

        if k < 0:
            return 0

        return λ * e ** (-1 * λ * k)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given number of “successes”
        Args:
            x (float): Number of “successes”

        Returns:
               cdf (float): The PMF value for k.
        """

        k = x
        λ = self.lambtha
        e = self.e

        if k < 0:
            return 0

        return 1 - e ** (-1 * λ * k)
