#!/usr/bin/env python3
"""
Module to work with milticlass classification
"""

import numpy as np


def one_hot_decode(one_hot):
    """
        Converts a  one-hot matrix into a vector of labels::
        Args:
            - one_hot is numpy.ndarray with shape (classes, m)
            - classes is the maximum number of classes
            - m is the number of examples
        Returns:
            a one-hot encoding of Y with shape (classes, m),
            None on failure
            len(one_hot.shape) != 2
    """
    if not isinstance(one_hot, np.ndarray):
        return None

    if len(one_hot) == 0:
        return None

    if len(one_hot.shape) != 2:
        return None

    X = np.where(one_hot.T)
    Y = np.array(X[1])

    return Y
