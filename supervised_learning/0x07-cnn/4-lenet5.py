#!/usr/bin/env python3
"""
Module used to
"""

import tensorflow as tf


def lenet5(x, y):
    """
    Function that builds a modified version of the LeNet-5 architecture using
    tensorflow:

    The model should consist of the following layers in order:
    -   1. Convolutional layer with 6 kernels of shape 5x5 with same padding
    -   2. Max pooling layer with kernels of shape 2x2 with 2x2 strides
    -   3. Convolutional layer with 16 kernels of shape 5x5 with valid padding
    -   4. Max pooling layer with kernels of shape 2x2 with 2x2 strides
    -   5. Fully connected layer with 120 nodes
    -   6. Fully connected layer with 84 nodes
    -   7. Fully connected softmax output layer with 10 nodes

    All layers requiring initialization should initialize their kernels with
    the he_normal initialization method:
        tf.contrib.layers.variance_scaling_initializer()
    All hidden layers requiring activation must use the relu activation funct

    Args:
        - x:        is a tf.placeholder of shape (m, 28, 28, 1) containing the
                    input images for the network
            - m:    is the number of images

        - y:        is a tf.placeholder of shape (m, 10) containing the one-hot
                    labels for the network
    Returns:
        - a tensor for the softmax activated output
        - a training operation that utilizes Adam optimization (with default
            hyperparameters)
        - a tensor for the loss of the netowrk
        - a tensor for the accuracy of the network
    """

    # exampla taken from : https://bit.ly/2WI57tC

    init_ = tf.contrib.layers.variance_scaling_initializer()
    # 1. Conv layer, 6 kernels of shape 5x5 with same padding (hidden)
    cv_lyr1 = tf.layers.Conv2D(filters=6,
                               kernel_size=(5, 5),
                               padding='same',
                               kernel_initializer=init_,
                               activation=tf.nn.relu)(x)

    # 2. Max pooling layer, with kernels of shape 2x2 & 2x2 strides (hidden)
    pool_lyr_2 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2))(cv_lyr1)

    # 3. Conv layer, 16 kernels of shape 5x5 with same padding (hidden)
    cv_lyr3 = tf.layers.Conv2D(filters=16,
                               kernel_size=(5, 5),
                               padding='valid',
                               kernel_initializer=init_,
                               activation=tf.nn.relu)(pool_lyr_2)

    # 4. Max pooling layer, with kernels of shape 2x2 & 2x2 strides (hidden)
    pool_lyr_4 = tf.layers.MaxPooling2D(pool_size=(2, 2),
                                        strides=(2, 2))(cv_lyr3)

    # 5. Fully connected layer with 120 nodes
    flatten5 = tf.layers.Flatten()(pool_lyr_4)
    fc_lyr_5 = tf.contrib.layers.fully_connected(inputs=flatten5,
                                                 num_outputs=120,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=init_)

    # 6. Fully connected layer with 84 nodes
    # no need of flatten
    fc_lyr_6 = tf.contrib.layers.fully_connected(inputs=fc_lyr_5,
                                                 num_outputs=84,
                                                 activation_fn=tf.nn.relu,
                                                 weights_initializer=init_)

    # 7. Fully connected softmax output layer with 10 nodes
    # no need of flatten
    sfmx_ = tf.contrib.layers.fully_connected(inputs=fc_lyr_6,
                                              num_outputs=10,
                                              activation_fn=None,
                                              weights_initializer=init_)

    sfmx_lyr = tf.nn.softmax(sfmx_)

    # https://steadforce.com/first-steps-tensorflow-part-3/
    losses = tf.losses.softmax_cross_entropy(onehot_labels=y, logits=sfmx_)
    predictions = tf.equal(tf.argmax(sfmx_lyr, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(predictions, tf.float32))

    # https://github.com/tensorflow/docs/blob/r1.12/
    # site/en/api_docs/python/tf/train/AdamOptimizer.md
    train_operation = tf.train.AdamOptimizer().minimize(losses)

    return (sfmx_lyr, train_operation, losses, accuracy)
