#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Function that builds a transition layer as described in Densely Connected
    Convolutional Networks:
    1. Your code should implement compression as used in DenseNet-C
    2. All weights should use he normal initialization
    3. All convolutions should be preceded by Batch Normalization and a
       rectified linear activation (ReLU), respectively

    Args:
        - X is the output from the previous layer
        - nb_filters is an integer representing the number of filters in X
        - compression is the compression factor for the transition layer

    Returns:
         The output of the transition layer and the number of filters within
         the output, respectively
    """

    new_filter = int(nb_filters * compression)

    # https://bit.ly/2JwVKVS

    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    X = K.layers.Conv2D(new_filter,
                        (1, 1),
                        kernel_initializer='he_normal',
                        padding='same')(X)

    X = K.layers.AveragePooling2D((2, 2), strides=(2, 2))(X)

    return X, new_filter
