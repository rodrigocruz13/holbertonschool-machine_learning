#!/usr/bin/env python3
"""
Module used to calculate sums
"""

def summation_i_squared(n):
    """
    function that calculates sum from i=1 to n of i^2:
    Args:
        n (int): is the stopping condition

    Returns:
        sum (int): for success,
        None (): In case n is not valid.
    """

    if type(n) is not int or n < 1:
        return None
    else:
        sum = 0
        for i in range(n + 1):
            sum += (i * i)
    return sum
