#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Function  that builds an identity block as described in Deep Residual
    Learning for Image Recognition (2015)

    1. All convolutions inside the block should be followed by batch
       normalization along the channels axis and a rectified linear activation
       (ReLU), respectively.
    2 All weights should use he normal initialization

    Args:
    - A_prev is the output from the previous layer
    - filters is a tuple or list containing F11, F3, F12, respectively:
      - F11 is the number of filters in the first 1x1 convolution
      - F3 is the number of filters in the 3x3 convolution
      - F12 is the number of filters in the second 1x1 convolution
    Returns:
        the activated output of the identity block
    """

    # https://bit.ly/39wqCQU

    # Obtaining the values of the filters
    F11, F3, F12 = filters

    # Save the input valu
    X_shortcut = A_prev

    init_ = K.initializers.he_normal()

    # First component of main path
    X = K.layers.Conv2D(filters=F11,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding="same",
                        kernel_initializer=init_)(A_prev)

    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation("relu")(X)

    # Second component of main path
    X = K.layers.Conv2D(filters=F3,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding="same",
                        kernel_initializer=init_)(X)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation("relu")(X)

    # Third component of main path
    X = K.layers.Conv2D(filters=F12,
                        kernel_size=(1, 1),
                        strides=(1, 1),
                        padding="same",
                        kernel_initializer=init_)(X)

    X = K.layers.BatchNormalization()(X)
    # Final step, Add shortcut value to main path, and pass it through a RELU
    # activation
    X = K.layers.Add()([X, X_shortcut])
    X = K.layers.Activation("relu")(X)
    return X
