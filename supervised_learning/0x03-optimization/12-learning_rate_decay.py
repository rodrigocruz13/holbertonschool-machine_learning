#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy
    Args:
        alpha:       (α) is the original learning rate
        decay_rate:  is the weight used to find the rate at which α will decay
        global_step: is the # of passes of gradient descent that have elapsed
        decay_step:  is the number of passes of gradient descent that should
                     occur before alpha is decayed further
    Notes:
        The learning rate decay should occur in a stepwise fashion
    Returns:
        The updated value for alpha
    """

    α = alpha

    # tf.train.RMSPropOptimizer
    # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/RMSPropOptimizer.md
    # Optimizer that implements the RMSProp algorithm.
    α1 = tf.train.inverse_time_decay(learning_rate=α,
                                     global_step=global_step,
                                     decay_steps=decay_step,
                                     decay_rate=decay_rate,
                                     staircase=True)

    # Args minimize(
    # - loss, A Tensor containing the value to minimize
    # ...
    # Returns:
    # An Operation that updates the variables in var_list. If global_step
    # was not None, that operation also increments global_step.

    return α1
