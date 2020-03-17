#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def predict(network, data, verbose=True):
    """
    Function that saves a model’s configuration in JSON format:
    Args:
        - network is the network model to test
        - data is the input data to test the model with
        - verbose is a boolean that determines if output should be printed
          during the testing process
    Returns:
        The prediction for the data
    """

    predictions = network.predict(data, verbose=verbose)
    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    return (predictions)
