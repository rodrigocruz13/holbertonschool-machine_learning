#!/usr/bin/env python3
"""
Module used to use matrix mult
"""
import numpy as np


def np_slice(matrix, axes={}):
    """
    function that slices a matrix along a specific axes:

    Arguments:
    - matrix    : numpy.ndarray
    - axes      : dictionary
                  Dictionary where the key is an axis to slice along and the
                  value is a tuple representing the slice to make along that
                  axis
                  You can assume that axes represents a valid slice
    Return      : The sliced matrix
    Hint        : https://docs.python.org/3/library/functions.html#slice
    """

    sliced = []
    num_keys = max(axes)
    n = num_keys + 1

    for i in range(n):
        if (i in axes.keys()):
            sliced.append(slice(*axes.get(i)))
        else:
            sliced.append(slice(None, None, None))

    return matrix[tuple(sliced)]
