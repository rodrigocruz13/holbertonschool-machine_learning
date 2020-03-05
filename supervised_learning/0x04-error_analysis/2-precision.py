#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix:
    Args:
        - confusion: a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
            - classes is the number of classes
    Returns:
        numpy.ndarray of shape (classes,) with sensitivity of each class
    """

    pre = []
    i = 0
    for row in confusion:
        positive = row[i]
        column = confusion.sum(axis=0)
        pre.append(positive / column[i])
        i = i + 1

    return np.array(pre)
