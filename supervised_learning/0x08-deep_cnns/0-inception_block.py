#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Function that builds an inception block as described in Going Deeper with
    Convolutions (2014)

    All convolutions inside the inception block should use a rectified linear
    activation (ReLU)

    Args:
        - A_prev:   is the output of the previous layer
        - filters   is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
                    respectively:
                -   F1 is the number of filters in the 1x1 convolution
                -   F3R is the number of filters in the 1x1 convolution before
                    the 3x3 convolution
                -   F3 is the number of filters in the 3x3 convolution
                -   F5R is the number of filters in the 1x1 convolution before
                    the 5x5 convolution
                -   F5 is the number of filters in the 5x5 convolution
                -   FPP is the number of filters in the 1x1 convolution after
                    the max pooling
    Returns:    the concatenated output of the inception block
    """

    # Obtaining the values of the filters
    F1, F3R, F3, F5R, F5, FPP = filters

    # https://bit.ly/2WU9VfJ

    # left column
    tower_0 = K.layers.Conv2D(filters=F1,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu")(A_prev)

    # left center column
    tower_1 = K.layers.Conv2D(filters=F3R,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu")(A_prev)

    tower_1 = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding="same",
                              activation="relu")(tower_1)

    # right center column
    tower_2 = K.layers.Conv2D(filters=F5R,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu")(A_prev)

    tower_2 = K.layers.Conv2D(filters=F5,
                              kernel_size=(5, 5),
                              padding="same",
                              activation="relu")(tower_2)
    # right column
    tower_3 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(1, 1),
                                    padding="same")(A_prev)

    tower_3 = K.layers.Conv2D(filters=FPP,
                              kernel_size=(1, 1),
                              padding="same",
                              activation="relu")(tower_3)

    output = K.layers.concatenate([tower_0, tower_1, tower_2, tower_3])

    return output
