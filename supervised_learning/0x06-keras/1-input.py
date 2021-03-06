#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a NN with the Keras library **WITHOUT** the sequential class
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

    # https://www.tensorflow.org/guide/keras/functional
    inputs = K.Input(shape=(nx,))
    n_layers = len(layers)
    regularizer = K.regularizers.l2(λ)

    # input layer creation
    outputs = K.layers.Dense(units=layers[0],
                             kernel_regularizer=regularizer,
                             activation=activations[0],
                             name="dense")(inputs)  # inputs

    for i in range(1, n_layers):
        dropout = K.layers.Dropout(1 - keep_prob)(outputs)
        outputs = K.layers.Dense(units=layers[i],
                                 kernel_regularizer=regularizer,
                                 activation=activations[i],
                                 input_dim=nx,
                                 name="dense_" + str(i))(dropout)  # old layer
    a_model = K.Model(inputs, outputs)
    return a_model
