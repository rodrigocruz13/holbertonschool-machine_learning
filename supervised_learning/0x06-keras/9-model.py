#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def save_model(network, filename):
    """
    Function that saves an entire model.
    Args:
        - network is the model to save
        - filename is the path of the file that the model should be saved to
    Returns:
        None
    """

    network.save(filename)

    return (None)


def load_model(filename):
    """
    Function that  loads an entire model
    Args:
        - filename is the path of the file that the model should be saved to
    Returns:
        the loaded model
    """

    return (K.models.load_model(filename))
