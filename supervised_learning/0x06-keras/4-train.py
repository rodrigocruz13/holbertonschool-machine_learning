#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def train_model(
        network,
        data,
        labels,
        batch_size,
        epochs,
        verbose=True,
        shuffle=False):
    """
     Function that trains a model using mini-batch gradient descent:
        Args:
        - network is the model to train
        - data is a numpy.ndarray of shape (m, nx) containing the input data
        - labels is a one-hot numpy.ndarray of shape (m, classes) containing
          the labels of data
        - batch_size is the size of the batch used for mini-batch gradient
          descent
        - epochs is the number of passes through data for mini-batch gradient
          descent
        - verbose is a boolean that determines if output should be printed
          during training
        - shuffle is a boolean that determines whether to shuffle the batches
          every epoch. Normally, it is a good idea to shuffle, but for
          reproducibility, we have chosen to set the default to False.
    Returns:
        the History object generated after training the model
    """

    # Train from NumPy data
    # For small datasets, use in-memory NumPy arrays to train and eval a model
    # The model is "fit" to the training data using the fit method:
    history = network.fit(data,
                          labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          verbose=verbose)
    return(history)
