#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in tensorflow
    using the gradient descent with momentum optimization algorithm:

    Args:
        loss: is the loss of the network
        alpha: is the learning rate
        beta1: is the momentum weight
    Returns:
        The momentum optimization operation
    """

    α = alpha
    β1 = beta1

    # tf.train.MomentumOptimizer
    # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/MomentumOptimizer.md
    # Optimizer that implements the Momentum algorithm.
    train = tf.train.MomentumOptimizer(learning_rate=α, momentum=β1)

    # Args minimize(
    # - loss, A Tensor containing the value to minimize
    # ...
    # Returns:
    # An Operation that updates the variables in var_list. If global_step
    # was not None, that operation also increments global_step.

    op = train.minimize(loss)
    return op
