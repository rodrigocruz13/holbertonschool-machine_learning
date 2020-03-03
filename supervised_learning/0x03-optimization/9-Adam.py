#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using the Adam optimization algorithm::

    Args:
        alpha:   is the learning rate
        beta1:   is the momentum weight
        beta2:   is the momentum weight
        epsilon: is a small number to avoid division by zero
        var:     is a numpy.ndarray containing the variable to be updated
        grad:    is a numpy.ndarray containing the gradient of var
        v:       is the previous first moment of var
        s:       is the previous second moment of var
        t:       is the time step used for bias correction
    Returns:
        The updated variable, the new first moment, and the new second moment,
        respectively
    """

    # Adam optimization
    # https://www.youtube.com/watch?v=JXQT_vxqwIs

    α = alpha
    β1 = beta1
    β2 = beta2
    ε = epsilon

    Vd = (β1 * v) + ((1 - β1) * grad)
    Sd = (β2 * s) + ((1 - β2) * grad * grad)

    Vd_ok = Vd / (1 - β1 ** t)
    Sd_ok = Sd / (1 - β2 ** t)
    
    w = var - α * (Vd_ok / ((Sd_ok ** (0.5)) + ε))
    return (w, Vd, Sd)
