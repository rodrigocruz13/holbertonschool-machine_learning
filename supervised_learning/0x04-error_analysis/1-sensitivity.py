#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def sensitivity(confusion):
    """
    calculates the normalization (standardization) constants of a matrix:
    Args:
        - confusion: a confusion numpy.ndarray of shape (classes, classes)
        where row indices represent the correct labels and column indices
        represent the predicted labels
            - classes is the number of classes
    Returns:
        numpy.ndarray of shape (classes,) with the sensitivity of each class
    """

    sensitivity = []
    i = 0
    for row in confusion:
        positive = row[i]
        false_positive = np.sum(row)
        sensitivity_i = round(positive / false_positive, 8)
        sensitivity.append(sensitivity_i)
        i = i + 1

    return (sensitivity)
