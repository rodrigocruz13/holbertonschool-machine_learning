#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def train_model(network,
                data,
                labels,
                batch_size,
                epochs,
                validation_data=None,
                early_stopping=False,
                patience=0,
                verbose=True,
                shuffle=False):
    """
    Function that trains a model using mini-batch gradient descent:
    * Update of the 4-train.py. to also analyze validaiton data:
       validation_data is the data to validate the model with, if not None
        Args:
        - network is the model to train
        - data is a numpy.ndarray of shape (m, nx) containing the input data
        - labels is a one-hot numpy.ndarray of shape (m, classes) containing
          the labels of data
        - batch_size is the size of the batch used for mini-batch gradient
          descent
        - epochs is the number of passes through data for mini-batch gradient
          descent
        - validation_data,
        - verbose,
        - shuffle
    Returns:
        the History object generated after training the model
    """

    # Train from NumPy data
    # For small datasets, use in-memory NumPy arrays to train and eval a model
    # The model is "fit" to the training data using the fit method:

    callbacks = []
    if (validation_data):
        early_stopping = K.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=patience)
        callbacks.append(early_stopping)

    history = network.fit(x=data,
                          y=labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          verbose=verbose,
                          shuffle=shuffle,
                          validation_data=validation_data,
                          callbacks=callbacks)
    return (history)
