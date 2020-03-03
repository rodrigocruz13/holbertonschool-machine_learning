#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm:

    Args:
        loss     is the loss of the network
        alpha:   is the learning rate
        beta1:   is the weight used for the first moment
        beta2:   is the momentum weight
        epsilon: is a small number to avoid division by zero
    Returns:
        the Adam optimization operation
    """

    α = alpha
    β1 = beta1
    β2 = beta2
    ε = epsilon

    # tf.train.RMSPropOptimizer
    # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/RMSPropOptimizer.md
    # Optimizer that implements the RMSProp algorithm.
    train = tf.train.AdamOptimizer(learning_rate=α, beta1=β1, beta2=β2, epsilon=ε)

    # Args minimize(
    # - loss, A Tensor containing the value to minimize
    # ...
    # Returns:
    # An Operation that updates the variables in var_list. If global_step
    # was not None, that operation also increments global_step.

    op = train.minimize(loss)
    return op
