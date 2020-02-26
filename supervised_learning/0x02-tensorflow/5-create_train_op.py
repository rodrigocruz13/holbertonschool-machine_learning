#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf


def create_train_op(loss, alpha):
    """
    function that  creates the training operation for the network:
    Args:
        - loss is the loss of the network’s prediction
        - alpha is the learning rate
    Returns:
        Returns: an operation that trains the network using gradient descent
    """

    α = alpha
    train = tf.train.GradientDescentOptimizer(α).minimize(loss)

    return train
