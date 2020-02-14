#!/usr/bin/env python3
"""
Class for the Normal distribution
"""


class Normal:
    """ Class """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialize Normal
        Args:
            - data (lst): List of the data to be used to estimate the dist
            - mean (float): the mean of the distribution
            - stddev (float): Standard deviation of the distribution
        Returns:
            Nothing. Just initialize the variables
        """
        self.mean = float(mean)
        μ = self.mean
        self.stddev = float(stddev)
        σ = self.stddev

        π = 3.1415926536
        e = 2.7182818285

        if data is None:
            if σ <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            elif len(data) < 2:
                raise ValueError("data must contain multiple values")

            n = len(data)
            μ = sum(data) / n
            self.mean = μ

            variance = sum((x - μ)**2 for x in data) / n
            σ = variance ** 0.5
            self.stddev = σ

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
