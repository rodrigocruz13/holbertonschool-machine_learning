#!/usr/bin/env python3
"""
Autoencoders
"""
import numpy as np
import tensorflow.keras as K


def sparse(input_dims, hidden_layers, latent_dims, lambtha):
    """
    Function that creates a sparce autoencoder:

    Arguments
    ---------
    - input_dims      : integer containing the dimensions of the model input
    - hidden_layers   : list containing the number of nodes for each hidden
                        layer in the encoder, respectively
                        Note : they should be reversed for the decoder
    - latent_dims     : integer containing the dimensions of the latent space
                        representation
    - lambtha         : is the regularization parameter used for L1
                        regularization on the encoded output

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
    encoded = x = K.Input(shape=(input_dims, ))
    regularizater = K.regularizers.l1(lambtha)

    for i in range(n):
        x = K.layers.Dense(hidden_layers[i],
                           activation="relu",
                           regularizater=regularizater)(x)
    h = K.layers.Dense(latent_dims,
                       activation="relu",
                       regularizater=regularizater)(x)
    encoder = K.models.Model(inputs=encoded, outputs=h)

    # 2. Decoder
    decoded = y = K.Input(shape=(latent_dims, ))
    # start = n-1, stop = -1, step = -1
    for j in range((n - 1), -1, -1):
        y = K.layers.Dense(hidden_layers[j],
                           activation="relu",
                           regularizater=regularizater)(y)
    r = K.layers.Dense(input_dims, activation="sigmoid")(y)
    decoder = K.models.Model(inputs=decoded, outputs=r)

    # 3. Autoencoder
    inputs = K.Input(shape=(input_dims, ))
    outputs = decoder(encoder(inputs))

    autoencoder = K.models.Model(inputs=inputs, outputs=outputs)
    autoencoder.compile(optimizer="adam", loss="binary_crossentropy")

    return encoder, decoder, autoencoder
