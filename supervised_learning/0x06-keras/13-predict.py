#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Function that makes a prediction using a neural network:
    Args:
        - network is the network model to test
        - data is the input data to test the model with
        - verbose is a boolean that determines if output should be printed
          during the testing process
    Returns:
        The prediction for the data
    """

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`

    return network.predict(data, verbose=verbose)

