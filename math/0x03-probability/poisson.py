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
            - data (lst): List of the data to be used to estimate the dist
            - lambtha (int): Expected number of events in a given time frame
        Returns:
            λ
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
            λ = float(sum(data) / len(data))
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

    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of “successes”
        Args:
            k (float): Number of “successes”

        Returns:
               PMF (float): The PMF value for k.
        """

        if k <= 0:
            return 0

        k = int(k)
        λ = self.lambtha
        k_f = self.factorial(k)
        e = self.e

        return (λ ** k) * (e ** -λ) / k_f

    def cdf(self, k):
        """
        Calculates the value of the CDF for a given number of “successes”
        Args:
            k (float): Number of “successes”

        Returns:
               CMF (float): The PMF value for k.
        """

        if k <= 0:
            return 0

        k = int(k)

        cdf = 0
        while (k > 0):
            cdf += self.pmf(k)
            k = k - 0.9999999999
        return cdf
