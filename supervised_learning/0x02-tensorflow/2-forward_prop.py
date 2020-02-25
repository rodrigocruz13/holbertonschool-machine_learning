#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    function that creates the forward propagation graph for the neural
    network:

    Args:
        - x is the placeholder for the input data
        - layer_sizes is a list containing the number of nodes in each
          layer of the network
        - activations is a list containing the activation functions for
          each layer of the network
    Returns:
        the prediction of the network in tensor form
    """

    # print(x) = Tensor("x:0", shape=(?, 784), dtype=float32)
    # print(layer_sizes) = [256, 256, 10]
    # print(activations) = [<function tanh at 0x7efe482730d0>,
    #                       <function tanh at 0x7efe482730d0>,
    #                       None]

    input_ = x
    i = 0

    while(i < len(layer_sizes)):
        layer_lenght_ = layer_sizes[i]
        layer_activation_ = activations[i]

        if (i != 0):
            input_ = new_layer

        new_layer = create_layer(input_, layer_lenght_, layer_activation_)
        i = i + 1

    return new_layer
