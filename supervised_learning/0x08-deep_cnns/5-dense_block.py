#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Function that builds a dense block as described in Densely Connected
    Convolutional Networks:
    1. You should use the bottleneck layers used for DenseNet-B
    2. All weights should use he normal initialization
    3. All convolutions should be preceded by Batch Normalization and a
       rectified linear activation (ReLU), respectively
    Args:
    - X is the output from the previous layer
    - nb_filters is an integer representing the number of filters in X
    - growth_rate is the growth rate for the dense block
    - layers is the number of layers in the dense block
    Returns:
        The concatenated output of each layer within the Dense Block and the
        number of filters within the concatenated outputs, respectively
    """

    # https://bit.ly/2JwVKVS

    for i in range(layers):
        each_layer = DenseNetB_layer(X, growth_rate)
        X = K.layers.concatenate([X, each_layer])
        nb_filters += growth_rate

    return X, nb_filters


def DenseNetB_layer(X, growth_rate):
    """
    Helper function that generates the DenseNet-B (Bottleneck Layers)

    Args:
    - X is the output from the previous layer
    - growth_rate is the growth rate for the dense block
    Returns:
        A layer with the Densenet-B layers applied
    """

    # 2.2. DenseNet-B (Bottleneck Layers)
    # To reduce the model complexity and size, BN-ReLU-1×1 Conv is done
    # before BN-ReLU-3×3 Conv.

    # 1x1
    gr = growth_rate * 4
    a_layer = K.layers.BatchNormalization()(X)
    a_layer = K.layers.Activation('relu')(a_layer)
    a_layer = K.layers.Conv2D(filters=gr,
                              kernel_size=(1, 1),
                              kernel_initializer='he_normal',
                              padding='same')(a_layer)

    # 3x3
    a_layer = K.layers.BatchNormalization()(a_layer)
    a_layer = K.layers.Activation('relu')(a_layer)
    a_layer = K.layers.Conv2D(filters=growth_rate,
                              kernel_size=(3, 3),
                              kernel_initializer='he_normal',
                              padding='same')(a_layer)

    return a_layer
