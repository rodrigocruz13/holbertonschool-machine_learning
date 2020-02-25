#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    function thatcalculates the softmax cross-entropy loss of a prediction:
    Args:
        - y is a placeholder for the labels of the input data
        - y_pred is a tensor containing the network’s predictions
    Returns:
        Returns: a tensor containing the loss of the prediction
    """

    # from Github documentation link: shorturl.at/dKTY3

    ŷ = y_pred
    loss_ = tf.losses.softmax_cross_entropy(y, ŷ)

    return loss_
