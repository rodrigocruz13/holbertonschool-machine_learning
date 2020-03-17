#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Builds a NN with the Keras library **WITHOUT** the sequential class
        Args:
        - network is the model to optimize
        - alpha is the learning rate
        - beta1 is the first Adam optimization parameter
        - beta2 is the second Adam optimization parameter
    Returns:
        None
    """

    α = alpha
    β1 = beta1
    β2 = beta2
    loss = "categorical_crossentropy"

    # optimizer
    opt = K.optimizers.Adam(α, β1, β2)

    # compiling
    network.compile(optimizer=opt,
                    loss=loss,
                    metrics=['accuracy'])
    return None
