#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    function that returns two placeholders, x and y, for the neural network:
    Args:
        nx (int): the number of feature columns in our data
        classes (int): the number of classes in our classifier

    Returns:
        placeholders named x and y, respectively
        x is the placeholder for the input data to the neural network
        y is the placeholder for the one-hot labels for the input data
    """

    x = tf.placeholder(float, shape=[None, nx], name="x")
    y = tf.placeholder(float, shape=[None, classes], name="y")

    return (x, y)


def create_layer(prev, n, activation):
    """
    function that returns  tensor output of the layer.
    Args:
        - prev is the tensor output of the previous layer
        - n is the number of nodes in the layer to create
        - activation is the activation function that the layer should use:
                - Use tf.contrib.layers.variance_scaling_initializer
                (mode="FAN_AVG") to implement He et. al initialization
                for the layer weights
                - each layer should be given the name layer
    Returns:
        the tensor output of the layer
    """

    raw_layer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    # (1)
    init = tf.global_variables_initializer()

    # (2)
    output_tensor = tf.layers.dense(inputs=prev, 
                                    units=n,
                                    activation=activation,
                                    kernel_initializer=raw_layer,
                                    name="layer")

    return(output_tensor)

    # (1)
    # The layer contains variables that must be initialized before they can be
    # used. While it is possible to initialize variables individually, you can
    # easily initialize all the variables in a TensorFlow graph as follows:
    # init = tf.global_variables_initializer()

    # Important: Calling tf.global_variables_initializer only creates and
    # returns a handle to a TensorFlow operation. That op will initialize all
    # the global variables when we run it with tf.Session.run.
    # Also note that this global_variables_initializer only initializes
    # variables that existed in the graph when the initializer was created.
    # So the initializer should be one of the last things added during graph
    # construction.

    # (2)
    # https://bit.ly/390xI0B   tf.layers.Dense
