#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Function that saves an entire model.
    Args:
        - network is the model to save
        - filename is the path of the file that the model should be saved to
        - save_format is the format in which the weights should be saved
    Returns:
        None
    """

    network.save_weights(filename, save_format='h5')
    return (None)


def load_weights(network, filename):
    """
    Function that  loads an entire model
    Args:
        - network is the model to which the weights should be loaded
        - filename is the path of the file that the model should be saved to
    Returns:
        None
    """

    network.load_weights(filename)
    return (None)
