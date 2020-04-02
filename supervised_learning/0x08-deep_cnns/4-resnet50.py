#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Function  that builds the ResNet-50 architecture as described in Deep
    Residual Learning for Image Recognition (2015):

    1. You can assume the input data will have shape (224, 224, 3)
    2. All convolutions inside and outside the blocks should be followed by
       batch normalization along the channels axis and a rectified linear
       activation (ReLU), respectively.
    3. All weights should use he normal initialization

    Returns:    the keras model
    """

    # https://bit.ly/2UVh8tr

    init_ = K.initializers.he_normal(seed=None)

    # Define the input as a tensor with shape input_shape
    i = K.layers.Input(shape=(224, 224, 3))

    # Stage 1
    X = K.layers.Conv2D(filters=64,
                        kernel_size=(7, 7),
                        padding='same',
                        strides=(2, 2),
                        kernel_initializer=init_)(i)

    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.MaxPooling2D((3, 3), padding='same', strides=(2, 2))(X)

    # Stage 2
    X = projection_block(X, filters=[64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    # Stage 3
    X = projection_block(X, filters=[128, 128, 512], s=2)
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])

    # Stage 4
    X = projection_block(X, filters=[256, 256, 1024], s=2)
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])

    # Stage 5
    X = projection_block(X, filters=[512, 512, 2048], s=2)
    X = identity_block(X, [512, 512, 2048])
    X = identity_block(X, [512, 512, 2048])

    # AVGPOOL
    X = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(X)

    # Output layer
    # X = K.layers.Flatten()(X)
    X = K.layers.Dense(1000, activation='softmax', kernel_initializer=init_)(X)

    # Create model
    model = K.models.Model(inputs=i, outputs=X, name='ResNet50')

    return model
