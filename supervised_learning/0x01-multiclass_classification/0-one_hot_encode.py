#!/usr/bin/env python3
"""
Module to work with milticlass classification
"""

import numpy as np


def one_hot_encode(Y, classes):
    """
        Converts a numeric label vector into a one-hot matrix:
        Args:
            - Y is a numpy.ndarray with shape (m, )
            containing numeric class labels
            - m is the number of examples

        Returns:
            a one-hot encoding of Y with shape (classes, m),
            None on failure
    """
    if isinstance(Y, np.ndarray):
        if (isinstance(classes, int) and classes > 0 and Y.shape[0] > 0):
            maximum = np.max(Y)
            if (classes <= Y.shape[0] and classes > maximum):
                # create empty matrix
                encoded_Y = np.zeros((classes, Y.shape[0]))

                # Hot encoding
                for i in range(Y.shape[0]):
                    encoded_Y[Y[i]][i] = 1
                return encoded_Y
    return None
