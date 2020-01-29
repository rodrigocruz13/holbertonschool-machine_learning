#!/usr/bin/env python3
"""
Module used to use concatenate matrices in an axis
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Matrix concatenate operations
    """
    return (np.concatenate((mat1, mat2), axis=axis))
