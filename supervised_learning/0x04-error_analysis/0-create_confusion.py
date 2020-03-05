#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix
    Args:
        - Labels is a one-hot numpy.ndarray of shape (m, classes)
          containing the correct labels for each data point
            - m is the number of data points
            - classes is the number of classes
        - logits is a one-hot numpy.ndarray of shape (m, classes)
          containing the predicted labels
    Returns:
        Mean and standard deviation of each feature, respectively
    """

    # labels.shape = 50.000 x 10
    # logits.shape = 50.0000 x 10
    # Answer = 10 x 10  --->then --->  labels.transpose

    confusion = np.dot(labels.T, logits)

    return (confusion)
