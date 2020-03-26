#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K


def lenet5(X):
    """
    Function that builds a modified version of the LeNet-5 architecture using
    keras:

    The model should consist of the following layers in order:
        1. Convolutional layer with 6 kernels of shape 5x5 with same padding
        2. Max pooling layer with kernels of shape 2x2 with 2x2 strides
        3. Convolutional layer with 16 kernels of shape 5x5 with valid padding
        4. Max pooling layer with kernels of shape 2x2 with 2x2 strides
        5. Fully connected layer with 120 nodes
        6. Fully connected layer with 84 nodes
        7. Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method

    All hidden layers requiring activation should use the relu activation
    function

    Args:
        - X:    is a K.Input of shape (m, 28, 28, 1) containing the input
                images for the network
                - m is the number of image
    Returns:
        a K.Model compiled to use Adam optimization (with default
        hyperparameters) and accuracy metrics
    """

    # exampla taken from : https://bit.ly/3bnQ10z

    init_ = K.initializers.he_normal()
    # 1. Conv layer, 6 kernels of shape 5x5 with same padding (hidden)
    cv_lyr1 = K.layers.Conv2D(filters=6,
                              kernel_size=(5, 5),
                              padding='same',
                              kernel_initializer=init_,
                              activation='relu')(X)

    # 2. Max pooling layer, with kernels of shape 2x2 & 2x2 strides (hidden)
    pool_lyr_2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                       strides=(2, 2))(cv_lyr1)

    # 3. Conv layer, 16 kernels of shape 5x5 with same padding (hidden)
    cv_lyr3 = K.layers.Conv2D(filters=16,
                              kernel_size=(5, 5),
                              padding='valid',
                              kernel_initializer=init_,
                              activation='relu')(pool_lyr_2)

    # 4. Max pooling layer, with kernels of shape 2x2 & 2x2 strides (hidden)
    pool_lyr_4 = K.layers.MaxPooling2D(pool_size=(2, 2),
                                       strides=(2, 2))(cv_lyr3)

    # 5. Fully connected layer with 120 nodes
    flatten5 = K.layers.Flatten()(pool_lyr_4)
    fc_lyr_5 = K.layers.Dense(units=120,
                              activation='relu',
                              use_bias=False,
                              kernel_initializer=init_)(flatten5)

    # 6. Fully connected layer with 84 nodes
    # no need of flatten
    fc_lyr_6 = K.layers.Dense(units=84,
                              activation='relu',
                              use_bias=False,
                              kernel_initializer=init_)(fc_lyr_5)

    # 7. Fully connected softmax output layer with 10 nodes
    # no need of flatten
    sfmx_lyr = K.layers.Dense(units=10,
                              activation='softmax',
                              use_bias=False,
                              kernel_initializer=init_)(fc_lyr_6)

    model = K.Model(inputs=X, outputs=sfmx_lyr)
    model.compile(optimizer='Adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return (model)
