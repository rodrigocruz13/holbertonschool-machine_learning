#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Function that builds an inception network as described in Going Deeper
    with Convolutions (2014)

    1. You can assume the input data will have shape (224, 224, 3)
    2. All convolutions inside and outside the inception block should use a
       rectified linear activation (ReLU)

    Returns:    the keras model
    """

    # https://www.youtube.com/watch?v=KfV8CJh7hE0

    input_image = K.layers.Input(shape=(224, 224, 3))  # 0

    # Layer 1
    x = K.layers.Conv2D(64, 7, strides=2, padding='same', activation='relu')(i)
    x = K.layers.MaxPool2D(3, strides=2, padding='same')(x)

    # Layer 2
    x = K.layers.Conv2D(64, 1, activation='relu')(x)
    x = K.layers.Conv2D(192, 3, padding='same', activation='relu')(x)
    # x = K.layers.MaxPool2D(3, strides=2)(x)
    x = K.layers.MaxPool2D(3, strides=2, padding='same')(x)

    # Layer 3
    x = inception_block(x, [64, 96, 128, 16, 32, 32])
    x = inception_block(x, [128, 128, 192, 32, 96, 64])
    x = K.layers.MaxPool2D(3, strides=2, padding='same')(x)

    # Layer 4
    x = inception_block(x, [192, 96, 208, 16, 48, 64])
    x = inception_block(x, [160, 112, 224, 24, 64, 64])
    x = inception_block(x, [128, 128, 256, 24, 64, 64])
    x = inception_block(x, [112, 144, 288, 32, 64, 64])
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = K.layers.MaxPool2D(3, strides=2, padding='same')(x)

    # Layer 5
    x = inception_block(x, [256, 160, 320, 32, 128, 128])
    x = inception_block(x, [384, 192, 384, 48, 128, 128])

    # Layer 6
    x = K.layers.AvgPool2D(7, strides=1)(x)
    x = K.layers.Dropout(0.4)(x)

    # x = K.layers.Flatten()(x)
    output = K.layers.Dense(1000, activation='softmax')(x)
    model = K.models.Model(input_image, output)

    return model
