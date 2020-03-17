#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization:
    Args:
        - cost is the cost of the network without L2 regularization

    Returns:
        a tensor containing the cost of the network accounting for
        L2 regularization
    """

    # Cost = Entrophy cost (or cost) + L2_cost
    # https://datascience-enthusiast.com/DL/Improving_DeepNeural_Networks_Regularization.html

    entrophy_cost = cost
    cost_l2 = tf.losses.get_regularization_losses(scope=None)
    # Gets the list of regularization losses.
    # Returns: A list of regularization losses as Tensors.

    return(entrophy_cost + cost_l2)
