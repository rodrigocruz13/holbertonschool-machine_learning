#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    function that that calculates the accuracy of a prediction:

    Args:
        - y is a placeholder for the labels of the input data
        - y_pred is a tensor containing the network’s predictions
    Returns:
        Returns: a tensor containing the decimal accuracy of the prediction
    """

    # from Stackoverflow: https://bit.ly/390AzXw
    # shorturl.at/mCF13

    ŷ = y_pred

    ŷ1 = tf.argmax(ŷ, 1)
    y1 = tf.argmax(y, 1)

    equality = tf.equal(ŷ1, y1)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return accuracy
