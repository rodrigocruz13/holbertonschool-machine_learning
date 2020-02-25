#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def calculate_accuracy(y, y_pred):
    """
    function that that calculates the accuracy of a prediction:

    Args:
        - y is a placeholder for the labels of the input data
        - y_pred is a tensor containing the network’s predictions
    Returns:
        Returns: a tensor containing the decimal accuracy of the prediction
    """

    ŷ = y_pred
    equality = tf.equal(ŷ, y)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy
