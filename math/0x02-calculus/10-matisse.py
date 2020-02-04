#!/usr/bin/env python3
"""
Module used to calculate derivates
"""

def poly_derivative(poly):
    """
    function that calculates a derivate of a poly
    Args:
        poly (lst): poly is a list of coefficients representing a polynomial
            The index is the power of x that the coefficient belongs to
            Example: if f(x) = x^3 + 3x +5, poly is equal to [5, 3, 0, 1]

    Returns:
        deriv (lst): list with the coefficient of de derivate
    """

    if type(poly) is not list:
        return None
    elif len(poly) == 0:
        return None
    elif len(poly) == 1:
        return [0]
    else:
        deriv = [0] * (len(poly) - 1)
        for i in range(len(poly) - 1):
            deriv[i] = poly[i + 1] * (i + 1)
    return deriv
