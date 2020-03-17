#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a NN with the Keras library **WITHOUT** the input class
    Args:
        - nx is the number of input features to the network
        - layers is a list containing the number of nodes in each layer of the
          network
        - activations is a list containing the activation functions used for
          each layer of the network
        - lambtha is the L2 regularization parameter
        - keep_prob is the probability that a node will be kept for dropout
    Returns:
        The keras model
    """
    λ = lambtha

    # create model
    a_model = K.Sequential()
    n_layers = len(layers)
    regularizer = K.regularizers.l2(λ)

    for i in range(n_layers):
        # Adds a densely-connected layer with layer[i] units to the model:
        a_model.add(K.layers.Dense(
            units=layers[i],
            input_dim=nx,
            kernel_regularizer=regularizer,
            activation=activations[i],
        )
        )
        # To avoid creation of:
        # Layer (type)            Output Shape      Param #
        # dropout_2 (Dropout)     (None, 10)        0
        if i < n_layers - 1:
            a_model.add(K.layers.Dropout(1 - keep_prob))
    return a_model
