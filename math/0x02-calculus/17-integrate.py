#!/usr/bin/env python3
"""
Module used to integrals
"""


def poly_integral(poly, C=0):
    """
    function that calculates the integral of a poly
    Args:
        poly (lst): poly is a list of coefficients representing a polynomial
            The index is the power of x that the coefficient belongs to
            Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]
        C (int): General constant
    Returns:
        inte (lst): list with the coefficient of the integral
    """

    if not isinstance(poly, list):
        return None

    if len(poly) == 0:
        return None

    if not isinstance(C, int):
        return None

    derivative = [C]
    if poly == [0]:
        return derivative

    if isinstance(poly, list):
        for i in range(len(poly)):
            if isinstance(poly[i], int) or isinstance(poly[i], float):
                val = poly[i] / (i + 1)
                derivative.append(int(val) if val.is_integer() else val)
            else:
                return None
        for summatory in range(len(derivative)):
            if (sum(derivative[summatory:]) is 0):
                return derivative[:summatory]
        return derivative
    else:
        return None
