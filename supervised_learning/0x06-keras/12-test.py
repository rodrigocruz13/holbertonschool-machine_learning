#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Function that saves a model’s configuration in JSON format:
    Args:
        - network is the network model to test
        - data is the input data to test the model with
        - labels are the correct one-hot labels of data
        - verbose is a boolean that determines if output should be printed
          during the testing process
    Returns:
        The loss & accuracy of the model with the testing data, respectively
    """

    results = network.evaluate(data, labels, verbose=verbose)
    # The returned "history" object holds a record of the loss values and
    # metric values during training
    return (results)
