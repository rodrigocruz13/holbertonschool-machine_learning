#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    Updates a variable using the RMSProp optimization algorithm:

    Args:
        loss     is the loss of the network
        alpha:   is the learning rate
        beta2:   is the RMSProp weight
        epsilon: is a small number to avoid division by zero
    Returns:
        the RMSProp optimization operation
    """

    α = alpha
    β2 = beta2
    ε = epsilon

    # tf.train.RMSPropOptimizer
    # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/RMSPropOptimizer.md
    # Optimizer that implements the RMSProp algorithm.

    # train=tf.train.RMSPropOptimizer(learning_rate=α, momentum=β2, epsilon=ε)
    # original and I dont undertand why decay = b2
    train = tf.train.RMSPropOptimizer(learning_rate=α, decay=β2, epsilon=ε)

    # Args minimize(
    # - loss, A Tensor containing the value to minimize
    # ...
    # Returns:
    # An Operation that updates the variables in var_list. If global_step
    # was not None, that operation also increments global_step.

    op = train.minimize(loss)
    return op
