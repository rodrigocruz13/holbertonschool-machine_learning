#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


# Factorial of a number using recursion

def fact(n):
    """
    Function that calculates the factorial of a positive integer:
    Args:
    - n     int                 number of patients that develop side effects
    Returns
            int                 factorial of n
    """

    if n == 1:
        return n
    else:
        return n * fact(n - 1)


def n_choose_x(n, x):
    """
    Function that calculates the factorial of a positive integer:
    Args:
    - n     int                 number of patients that develop side effects
    - x     int
    Returns
            int                 _ ways to choose an (unordered) subset of x
                                elements from a fixed set of n elements
    """

    n_fact = fact(n)
    x_fact = fact(x)
    nx_fact = fact(n - x)
    return n_fact / (x_fact * nx_fact)


def likelihood(x, n, P):
    """
    Function that calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects:

    Args:
    - x     int                 number of patients that develop side effects
    - n     int                 total number of patients observed
    - p     1D numpy.ndarray    Array with the various hypothetical
                                probabilities of developing side effects
    Note:
        - If n is not a positive integer, raise a ValueError with the message
        n must be a positive integer
        - If x is not an integer that is greater than or equal to 0, raise a
        ValueError with the message x must be an integer that is greater than
        or equal to 0
        - If x is greater than n, raise a ValueError with the message x cannot
        be greater than n
        - If P is not a 1D numpy.ndarray, raise a TypeError with the messag
        P must be a 1D numpy.ndarray
        - If any value in P is not in the range [0, 1], raise a ValueError
        with the message All values in P must be in the range [0, 1]
    Returns:    A 1D numpy.ndarray containing the likelihood of obtaining the
                data, x and n, for each probability in P, respectively
    """

    msg1 = "x must be an integer that is greater than or equal to 0"

    if not isinstance(n, int):
        raise ValueError("n must be a positive integer")

    if (n <= 0):
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (float, int)):
        raise ValueError(msg1)

    if (x < 0):
        raise ValueError(msg1)

    if (x > n):
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray):
        raise TypeError("P must be a 1D numpy.ndarray")

    if (len(P.shape) != 1):
        raise TypeError("P must be a 1D numpy.ndarray")

    if (np.any(P < 0) or np.any(P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    binomial_coef = n_choose_x(n, x)
    success_rate = pow(P, x)
    failure_rate = pow(1 - P, n - x)

    likelihood = binomial_coef * success_rate * failure_rate

    return likelihood
