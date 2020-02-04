#!/usr/bin/env python3


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

    if type(poly) is not list:
        return None
    if type(poly) is None:
        return None
    if type(C) is None or type(C) is int or type(C) is float:
        if len(poly) == 0:
            return [C]
        else:
            inte = [0] * (len(poly) + 1)
            for i in range(len(poly)):
                temp = poly[i] / (i + 1)
                if temp % 1 != 0:
                    inte[i + 1] = float(poly[i])/(i + 1)
                else:
                    inte[i + 1] = int((poly[i])/(i + 1))
            inte[0] = C
        return inte
    return None
