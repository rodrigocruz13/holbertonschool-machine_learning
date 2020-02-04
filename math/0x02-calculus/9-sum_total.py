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

    sum = 0

    if n is None or type(n) is not int or n < 1:
        return None
    else:
        return sum_recursive(n)


def sum_recursive(n):
    """
    function that calculates sum recursevily
    Args:
        n (int): is the stopping condition

    Returns:
        sum (int): sum
    """

    if n < 1:
        return 0
    else:
        return (n * n) + sum_recursive(n - 1)
