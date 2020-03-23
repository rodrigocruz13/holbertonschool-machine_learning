#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding='same', stride=(1, 1)):
    """
    Function that performs forward propagation over a convolutional layer of
    a neural network:
    Args:
        - A_prev is a numpy.ndarray with shape (m, h_prev, w_prev, c_prev)
          containing the output of the previous layer
            - m  is the number of examples
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
            - c_prev is the number of channels in the previous layer
        - W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
          kernels for the convolution
            - kh is the filter height
            - kw is the filter width
            - c_prev is the number of channels in the previous layer
            - c_new is the number of channels in the output
        - b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
          applied to the convolution
        - activation is an activation function applied to the convolution
        - padding is a string that is either same or valid, indicating the
          type of padding used
        - stride is a tuple of (sh, sw) containing the strides for the
          convolution
            - sh is the stride for the height
            - sw is the stride for the width
    Returns:
        the output of the convolutional layer
    """

    # num examples
    n_examples = A_prev.shape[0]

    # width and height of the previous layer
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]

    # kernel_width and kernel_height
    k_h = W.shape[0]
    k_w = W.shape[1]

    # channels of the previous layer and the new layer
    c_prev = W.shape[2]
    c_output = W.shape[3]

    # stride_height and stride_width
    s_h = stride[0]
    s_w = stride[1]

    # output_height and output_width
    o_h = h_prev - k_h + 1
    o_w = w_prev - k_w + 1
    if (padding == "same"):
        o_h = h_prev / s_h
        o_w = w_prev / s_h

    # creating outputs of size: [n_examples,  o_h  ⊛  o_w  ⊛  c_output]
    outputs = np.zeros((n_examples, o_h, o_w, c_output))

    # vectorizing the n_images into an array (creating a new dimension)
    imgs_arr = np.arange(0, n_examples)

    # iterating over the output array and generating the convolution
    for x in range(o_h):
        for y in range(o_w):
            for z in range(c_output):
                x0 = x * s_h
                y0 = y * s_w
                x1 = x0 + k_h
                y1 = y0 + k_w
                outputs[imgs_arr, x, y, z] = np.sum(np.multiply(
                    A_prev[imgs_arr, x0: x1, y0: y1], W[:, :, :, z]), axis=(1, 2, 3))

    return outputs