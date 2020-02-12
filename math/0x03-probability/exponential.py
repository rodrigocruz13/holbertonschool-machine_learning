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
        self.lambtha = lambtha
        self.π = 3.1415926536
        self.e = 2.7182818285

        λ = float(lambtha)

        if data is None:
            if λ <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")

            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            λ = 1 / float(sum(data) / len(data))
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
        root_π = π**(1/2)
        x3 = x**3
        x5 = x**5
        x7 = x**7
        x9 = x**9

        return (2/root_π) * x - x3/3 + x5/10 - x7/42 + x9/216

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
            x (float): Number of “successes”

        Returns:
               pdf (float): The PDF value for k.
        """

        if x < 0:
            return 0

        k = int(x)
        λ = self.lambtha
        e = self.e

        return λ * (e ** -λ * k)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given number of “successes”
        Args:
            x (float): Number of “successes”

        Returns:
               cdf (float): The PMF value for k.
        """

        if x <= 0:
            return 0

        k = int(x)

        cdf = 0
        while (k > 0):
            cdf += self.pdf(k)
            k = k - 0.9999999999
        return cdf
