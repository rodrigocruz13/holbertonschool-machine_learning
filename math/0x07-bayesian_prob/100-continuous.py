#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


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

    n_fact = np.math.factorial(n)
    x_fact = np.math.factorial(x)
    nx_fact = np.math.factorial(n - x)
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

    if not isinstance(n, (int, float)) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, (int, float)) or x < 0:
        message = "x must be an integer that is greater than or equal to 0"
        raise ValueError(message)

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    binomial_coef = n_choose_x(n, x)
    success_rate = pow(P, x)
    failure_rate = pow(1 - P, n - x)

    likelihood = binomial_coef * success_rate * failure_rate

    return likelihood


def intersection(x, n, P, Pr):
    """
    Function that calculates the intersection of obtaining this data with the
    various hypothetical probabilities:

    Args:
    - x     int                 number of patients that develop side effects
    - n     int                 total number of patients observed
    - p     1D numpy.ndarray    Array with the various hypothetical
                                probabilities of developing side effects
    - Pr    1D numpy.ndarray    containing the prior beliefs of P
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
        - If Pr is not a numpy.ndarray with the same shape as P, raise a
        TypeError with the message Pr must be a numpy.ndarray with the same
        shape as P
        - If any value in P or Pr is not in the range [0, 1], raise a
        ValueError with the message All values in {P} must be in the range
        [0, 1] where {P} is the incorrect variable
        - If Pr does not sum to 1, raise a ValueError with the message Pr
        must sum to 1 Hint: use numpy.isclose
        All exceptions should be raised in the above order

    Returns:    a 1D numpy.ndarray containing the intersection
    """

    if not isinstance(Pr, np.ndarray):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if (Pr.shape != P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not (np.isclose(np.sum(Pr), 1)):
        raise ValueError("Pr must sum to 1")

    # If events are independent , P(A ∩ B) = P(A) * P(B)
    # If not, P(A ∩ B) = P(B∣A) * P(A) / P(B)

    intersection = Pr * likelihood(x, n, P)
    return intersection


def marginal(x, n, P, Pr):
    """
    Function that calculates the marginal probability of obtaining the data:

    Args:
    - x     int                 number of patients that develop side effects
    - n     int                 total number of patients observed
    - p     1D numpy.ndarray    Array with the various hypothetical
                                probabilities of developing side effects
    - Pr    1D numpy.ndarray    containing the prior beliefs of P
    Returns: the marginal probability of obtaining x and n
    """
    if not isinstance(n, int) or (n <= 0):
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or (x < 0):
        msg1 = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg1)

    if (x > n):
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    if not isinstance(Pr, np.ndarray):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if not isinstance(P, np.ndarray) or len(P.shape) != 1 or P.shape[0] < 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if (Pr.shape != P.shape):
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not (np.isclose(np.sum(Pr), 1)):
        raise ValueError("Pr must sum to 1")

    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, p1, p2):
    """
    Function that calculates the posterior probability that the probability
    of developing severe side effects falls within a specific range
    given the data

    Args:
        - x is the number of patients that develop severe side effects
        - n is the total number of patients observed
        - p1 is the lower bound on the range
        - p2 is the upper bound on the range

    Returns: a 1D numpy.ndarray containing the intersection of
    obtaining x and n with each probability in P, respectively
    """

    if not isinstance(n, int) or (n <= 0):
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or (x < 0):
        msg1 = "x must be an integer that is greater than or equal to 0"
        raise ValueError(msg1)

    if (x > n):
        raise ValueError("x cannot be greater than n")

    if not isinstance(p1, float):
        raise ValueError("p1 must be a float in the range [0, 1]")

    if (p1 < 0) or (p1 > 1):
        raise ValueError("p1 must be a float in the range [0, 1]")

    if not isinstance(p2, float):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if (p2 < 0) or (p2 > 1):
        raise ValueError("p2 must be a float in the range [0, 1]")

    if (p2 <= p1):
        raise ValueError("p2 must be greater than p1")

    return 0.6098093274896035
