#!/usr/bin/env python3
"""
Autoencoders
"""
import numpy as np
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    Function that creates a convolutional autoencoder:

    Arguments
    ---------
    - input_dims      : tuple of integer containing the dimensions of the
                        model input
    - filters         : list containing the number of filters for each
                        convolutional layer in the encoder, respectively
                        Note: the filters should be reversed for the decoder
    - latent_dims     : tuple of integers containing the dimensions of the
                        latent space representation

    Notes:
            - Each convolution in the encoder should use a kernel size of
              (3, 3) with same padding and relu activation, followed by max
              pooling of size (2, 2)
            - Each convolution in the decoder, except for the last two, should
              use a filter size of (3, 3) with same padding and relu activation
              followed by upsampling of size (2, 2)
            - The second to last convolution should instead use valid padding
            - The last convolution should have the same number of filters as
              the number of channels in input_dims with sigmoid activation and
              no upsampling
    Returns
    -------
    - encoder           encoder model
    - decoder           decoder model
    - auto              full autoencoder model

    The autoencoder model should be compiled using adam optimization and
    binary cross-entropy loss
    All layers should use a relu activation except for the last layer in the
    decoder, which should use a sigmoid function
    """

    # https://towardsdatascience.com/deep-inside-autoencoders-7e41f319999f
    # https://blog.keras.io/building-autoencoders-in-keras.html

    input_encoder = keras.Input(shape=input_dims)
    kernel_size = 3
    pool_size = (2, 2)

    # 1. Encoder
    # Compressing the input to the botneckle
    encoded = keras.layers.Conv2D(filters=filters[0],
                                  kernel_size=kernel_size,
                                  padding='same',
                                  activation='relu')(input_encoder)
    encoded_pool = keras.layers.MaxPool2D(pool_size=pool_size,
                                          padding='same')(encoded)

    for i in range(1, len(filters)):
        encoded = keras.layers.Conv2D(filters=filters[i],
                                      kernel_size=kernel_size,
                                      padding='same',
                                      activation='relu')(encoded_pool)
        encoded_pool = keras.layers.MaxPool2D(pool_size=pool_size,
                                              padding='same')(encoded)

    latent = encoded_pool
    encoder = keras.models.Model(input_encoder, latent)
    encoder.summary()

    # decoded part of the model
    input_decoder = keras.Input(shape=(latent_dims))
    decoded = keras.layers.Conv2D(filters=filters[i],
                                  kernel_size=kernel_size,
                                  padding='same',
                                  activation='relu')(input_decoder)
    decoded_pool = keras.layers.UpSampling2D(size=[2, 2])(decoded)

    m = len(filters) - 2
    for i in range(m, -1, -1):  # start = m, stop = -1, step = -1
        padding = 'valid' if i == 0 else 'same'
        decoded = keras.layers.Conv2D(filters=filters[i],
                                      padding=padding,
                                      kernel_size=kernel_size,
                                      activation='relu')(decoded_pool)
        decoded_pool = keras.layers.UpSampling2D(size=[2, 2])(decoded)
    decoded = keras.layers.Conv2D(filters=1,
                                  padding=padding,
                                  kernel_size=kernel_size,
                                  activation='sigmoid')(decoded_pool)

    # decoder: mapp the input to reconstruct the original image
    decoder = keras.models.Model(input_decoder, decoded)
    decoder.summary()

    # 3. Autoencoder
    inputs = keras.Input(shape=(input_dims))
    outputs = decoder(encoder(inputs))

    # mapping the complete autoencoded model, reconstruc the image
    autoencoder = keras.models.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer="Adam", loss="binary_crossentropy")

    return encoder, decoder, autoencoder
