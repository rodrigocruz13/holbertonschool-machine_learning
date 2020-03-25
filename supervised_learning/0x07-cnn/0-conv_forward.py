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
        - A_prev:   is a numpy.ndarray with shape (m, h_prev, w_prev, c_prev)
                    containing the output of the previous layer
                    - m  is the number of examples
                    - h_prev is the height of the previous layer
                    - w_prev is the width of the previous layer
                    - c_prev is the number of channels in the previous layer
        - W:        is a numpy.ndarray of shape (kh, kw, c_prev, c_new)
                    containing the kernels for the convolution
                    - kh is the filter height
                    - kw is the filter width
                    - c_prev is the number of channels in the previous layer
                    - c_new is the number of channels in the output
        - b:        is a numpy.ndarray of shape (1, 1, 1, c_new) containing
                    the biases applied to the convolution
        - activation: is an activation function applied to the convolution
        - padding:  is a string that is either same or valid, indicating the
                    type of padding used
        - stride:   is a tuple of (sh, sw) containing the strides for the
                    convolution
                    - sh is the stride for the height
                    - sw is the stride for the width
    Returns:
        the output of the convolutional layer
    """

    # num examples
    m = A_prev.shape[0]

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

    # pad_h and pad_w ⊛
    p_h = 0
    p_w = 0

    if (padding == "same"):
        p_h = np.ceil(((s_h * h_prev) - s_h + k_h - h_prev) / 2)
        p_h = int(p_h)
        p_w = np.ceil(((s_w * w_prev) - s_w + k_w - w_prev) / 2)
        p_w = int(p_w)

    # output_height and output_width
    o_h = int((h_prev + 2 * p_h - k_h) / s_h) + 1
    o_w = int((w_prev + 2 * p_w - k_w) / s_w) + 1

    # creating outputs of size: [m,  o_h  ⊛  o_w  ⊛  c_output]
    outputs = np.zeros((m, o_h, o_w, c_output))

    # creating pad of zeros around the output images
    padded_imgs = np.pad(A_prev,
                         ((0, 0),       # dim n_images
                          (p_h, p_h),   # dim height
                          (p_w, p_w),   # dim width
                          (0, 0)        # dim channels
                          ),
                         mode="constant",
                         constant_values=0)

    # vectorizing the n_images into an array (creating a new dimension)
    imgs_arr = np.arange(0, m)

    # iterating over the output array and generating the convolution
    for x in range(o_h):
        for y in range(o_w):
            for z in range(c_output):
                x0 = x * s_h
                y0 = y * s_w
                x1 = x0 + k_h
                y1 = y0 + k_w

                A_ = padded_imgs[imgs_arr, x0: x1, y0: y1]
                W_ = W[:, :, :, z]
                b_ = b[:, :, :, z]

                mx = np.sum((A_ * W_), axis=(1, 2, 3))
                outputs[imgs_arr, x, y, z] = activation(mx + b_)

    return (outputs)
