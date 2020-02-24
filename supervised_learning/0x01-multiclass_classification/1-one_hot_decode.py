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

    if isinstance(one_hot, np.ndarray):
        if (one_hot.shape[0] > 0 and isinstance(one_hot.shape[0], int)):
            if (np.max(one_hot) > 0):
                lenght_ = one_hot.shape[1]
                a_list = []

                for i in range(lenght_):
                    column = one_hot[:, i]
                    key = np.unique(column, return_index=True, axis=None)
                    a_list.append(key[1][1])

                one_hot_decode = np.asarray(a_list)
            return (one_hot_decode)
    return None
