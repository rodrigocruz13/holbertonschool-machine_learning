#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def save_config(network, filename):
    """
    Function that saves a model’s configuration in JSON format:
    Args:
        - network is the model to save
        - filename is the path of the file that the model should be saved to
        - save_format is the format in which the weights should be saved
    Returns:
        None
    """

    json_model = network.to_json()
    with open(filename, 'w') as a_file:
        a_file.write(json_model)

    return (None)


def load_config(filename):
    """
    Function that loads an entire model with a specific configuration
    Args:
        - filename is the path of the file containing the model’s
          configuration in JSON format
    Returns:
        the loaded model
    """

    with open(filename, 'r') as a_file:
        a_model = K.models.model_from_json(a_file.read())
        return (a_model)
