#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Function that builds the DenseNet-121 architecture as described in Densely
    Connected Convolutional Networks:

    1. You can assume the input data will have shape (224, 224, 3)
    2. All convolutions should be preceded by Batch Normalization and a
       rectified linear activation (ReLU), respectively
    3. All weights should use he normal initialization

    Args:
        - growth_rate is the growth rate
        - compression is the compression factor

    Returns:
        The keras model
    """

    # https://github.com/sivaramakrishnan-rajaraman/CNN-for-malaria-parasite-detection/blob/master/densenet121_models.py

    # 1. You can assume the input data will have shape (224, 224, 3)
    i = K.Input(shape=(224, 224, 3))

    init_ = K.initializers.he_normal(seed=None)

    # 2. All convolutions should be preceded by Batch Normalization and a
    # rectified linear activation (ReLU), respectively

    x = K.layers.BatchNormalization()(i)
    x = K.layers.Activation("relu")(x)
    x = K.layers.Conv2D(filters=64,
                        kernel_size=(7, 7),
                        padding="same",
                        strides=(2, 2),
                        kernel_initializer=init_)(x)

    x = K.layers.MaxPooling2D(pool_size=(3, 3),
                              strides=(2, 2),
                              padding="same")(x)

    x, nf = dense_block(x, 64, growth_rate, 6)
    x, nf = transition_layer(x, nf, compression)
    x, nf = dense_block(x, nf, growth_rate, 12)
    x, nf = transition_layer(x, nf, compression)
    x, nf = dense_block(x, nf, growth_rate, 24)
    x, nf = transition_layer(x, nf, compression)
    x = dense_block(x, nf, growth_rate, 16)[0]

    x = K.layers.AveragePooling2D(pool_size=(7, 7),
                                  strides=None,
                                  padding="same")(x)
    x = K.layers.Dense(units=1000, activation="softmax")(x)

    model = K.models.Model(inputs=i, outputs=x)

    return model
