#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix:
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
        false_positive = sum(row)
        sensitivity.append(positive / false_positive)
        i = i + 1

    return np.array(sensitivity)
