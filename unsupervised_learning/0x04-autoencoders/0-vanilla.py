#!/usr/bin/env python3
"""
Autoencoders
"""

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    Function that creates an autoencoder:

    Arguments
    ---------
    - input_dims      : integer containing the dimensions of the model input
    - hidden_layers   : list containing the number of nodes for each hidden
                        layer in the encoder, respectively
                        Note : they should be reversed for the decoder
    - latent_dims     : integer containing the dimensions of the latent space
                        representation

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

    n = len(hidden_layers)
    # 1. Encoder
    encoded = x = keras.Input(shape=(input_dims, ))
    for i in range(n):
        x = keras.layers.Dense(hidden_layers[i], activation="relu")(x)
    h = keras.layers.Dense(latent_dims, activation="relu")(x)
    encoder = keras.models.Model(inputs=encoded, outputs=h)

    # 2. Decoder
    decoded = y = keras.Input(shape=(latent_dims, ))
    # start = n-1, stop = -1, step = -1
    for j in range((n - 1), -1, -1):
        y = keras.layers.Dense(hidden_layers[j], activation="relu")(y)
    r = keras.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = keras.models.Model(inputs=decoded, outputs=r)

    # 3. Autoencoder
    inputs = keras.Input(shape=(input_dims, ))
    outputs = decoder(encoder(inputs))

    autoencoder = keras.models.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, autoencoder
