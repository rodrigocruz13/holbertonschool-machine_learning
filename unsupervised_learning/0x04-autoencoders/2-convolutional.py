#!/usr/bin/env python3
"""
Autoencoders
"""

import tensorflow.keras as K
from tensorflow.keras.layers import Input, Dense, Conv2D
from tensorflow.keras.layers import MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers


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

    # https://www.datatechnotes.com/2020/03/convolutional-autoencoder-example-with-keras-in-python.html
    # https://blog.keras.io/building-autoencoders-in-keras.html

    input_img = K.layers.Input(shape=input_dims)

    enc_conv1 = Conv2D(12, (3, 3), activation='relu',
                       padding='same')(input_img)
    enc_pool1 = MaxPooling2D((2, 2), padding='same')(enc_conv1)
    enc_conv2 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_pool1)
    enc_ouput = MaxPooling2D((4, 4), padding='same')(enc_conv2)

    encoder = K.models.Model(input_img, enc_ouput)

    input_dec = K.layers.Input(shape=latent_dims)

    dec_conv2 = Conv2D(8, (4, 4), activation='relu', padding='same')(enc_ouput)
    dec_upsample2 = UpSampling2D((4, 4))(dec_conv2)
    dec_conv3 = Conv2D(12, (3, 3), activation='relu')(dec_upsample2)
    dec_upsample3 = UpSampling2D((2, 2))(dec_conv3)
    dec_output = Conv2D(1, (3, 3), activation='sigmoid',
                        padding='same')(dec_upsample3)

    decoder = K.models.Model(input_img, dec_output)

    autoencoder = Model(input_img, dec_output)
    autoencoder.compile(optimizer='Adam', loss='binary_crossentropy')

    return encoder, decoder, autoencoder
