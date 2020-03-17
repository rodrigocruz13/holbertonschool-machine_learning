#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Builds a NN with the Keras library **WITHOUT** the sequential class
        Args:
        - The last dimension of the one-hot matrix must be the number of classes
    Returns:
        the one-hot matrix
    """

    one_hot_ = K.utils.to_categorical(labels, classes)
    return(one_hot_)