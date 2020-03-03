#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using the gradient descent with momentum optimization:

    Args:
        alpha: is the learning rate
        beta1: is the momentum weight
        var:   is a numpy.ndarray containing the variable to be updated
        grad:  is a numpy.ndarray containing the gradient of var
        v:     is the previous first moment of var
    Returns:
        The updated variable and the new moment, respectively
    """

    # Gradient descent with momentum
    # https://www.youtube.com/watch?v=k8fTYJPd3_I

    α = alpha
    β1 = beta1
    dw = grad
    w = var

    v_dw = β1 * v + (1 - β1) * dw
    W = w - (α * v_dw)

    return W, v_dw
