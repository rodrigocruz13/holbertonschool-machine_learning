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

    # tf.train.inverse_time_decay
    # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/train/RMSPropOptimizer.md
    # Optimizer that implements the RMSProp algorithm.
    α1 = tf.train.inverse_time_decay(learning_rate=α,
                                     global_step=global_step,
                                     decay_steps=decay_step,
                                     decay_rate=decay_rate,
                                     staircase=True)

    # Args
    # - learning_rate: A scalar float32 or float64 Tensor or a Python number.
    #                  The initial learning rate.
    # - global_step:   A Python number. Global step to use for the decay
    #                  computation. Must not be negative.
    # - decay_steps:   How often to apply decay.
    # - decay_rate:    A Python number. The decay rate.
    # - staircase:     Whether to apply decay in a discrete staircase,
    #                  as opposed to continuous, fashion.
    # - name:          String. Optional name of the operation.
    #                  Defaults to 'InverseTimeDecay'.
    # Returns:
    # the learning rate decay operation

    return α1
