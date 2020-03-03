#!/usr/bin/env python3
"""
Module used to
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using the RMSProp optimization algorithm:

    Args:
        alpha:   is the learning rate
        beta2:   is the momentum weight
        epsilon: is a small number to avoid division by zero
        var:     is a numpy.ndarray containing the variable to be updated
        grad:    is a numpy.ndarray containing the gradient of var
        s:       is the previous second moment of var
    Returns:
        The updated variable and the new moment, respectively
    """

    α = alpha
    β2 = beta2
    ε = epsilon

    α = alpha
    dw = grad
    w = var

    s_new = β2 * s + (1 - β2) * (dw * dw)
    W = w - α * (dw / ((s_new ** 0.5) + ε))

    return W, s_new
