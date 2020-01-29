#!/usr/bin/env python3
"""
Module used to use concatenate matrices in an axis
"""


def np_cat(mat1, mat2, axis=0):
    """Matrix concatenate operations
    """

    import numpy as np

    return (np.concatenate((mat1, mat2), axis=axis))
