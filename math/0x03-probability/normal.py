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

        μ = self.mean
        σ = self.stddev
        π = 3.1415926536
        e = 2.7182818285

        exp = -1 * ((x - μ) ** 2) / (2 * (σ ** 2))
        den = 2 * π * (σ ** 2)

        pdf = (1 / (den) ** 0.5) * (e ** exp)

        return pdf

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given number of “successes”
        Args:
            x (float): Number of “successes”

        Returns:
               cdf (float): The PMF value for k.
        """

        μ = self.mean
        σ = self.stddev
        π = 3.1415926536

        x1 = (x - μ) / (σ * (2 ** 0.5))

        erf1 = 2 / π ** 0.5
        erf2 = (x1 -
                ((x1 ** 3) / 3) +
                ((x1 ** 5) / 10) -
                ((x1 ** 7) / 42) +
                ((x1 ** 9) / 216))
        erf = erf1 * erf2
        cdf = (1 + erf) / 2

        return cdf
