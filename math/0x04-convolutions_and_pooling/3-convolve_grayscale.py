#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a valid convolution on grayscale images
    Args:
        - images is a numpy.ndarray with shape (m, h, w) containing
          multiple grayscale images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
        - kernel is a numpy.ndarray with shape (kh, kw) containing the kernel
          for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
        - padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
            - if ‘same’, performs a same convolution
            - if ‘valid’, performs a valid convolution
            - if a tuple:
                - ph is the padding for the height of the image
                - pw is the padding for the width of the image
            the image should be padded with 0’s
        - stride is a tuple of (sh, sw)
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image
    Returns:
        A numpy.ndarray containing the convolved images
    """

    # num images
    n_images = images.shape[0]

    # input_width and input_height
    i_h = images.shape[1]
    i_w = images.shape[2]

    # kernel_width and kernel_height
    k_h = kernel.shape[0]
    k_w = kernel.shape[1]

    # stride_height and stride_width
    s_h = stride[0]
    s_w = stride[1]

    # pad_h and pad_w ⊛
    p_h = 0
    p_w = 0

    if (padding == "same"):
        p_h = int(((i_h - 1) * s_h + k_h - i_h) / 2) + 1
        p_w = int(((i_w - 1) * s_w + k_w - i_w) / 2) + 1

    elif (isinstance(padding, tuple)):
        p_h = padding[0]
        p_w = padding[1]

    # output_height and output_width
    o_h = np.floor(((i_h + 2 * p_h - k_h) / s_h) + 1).astype(int)
    o_w = np.floor(((i_w + 2 * p_w - k_w) / s_w) + 1).astype(int)

    # creating outputs of size: n_images, o_h x o_w
    outputs = np.zeros((n_images, o_h, o_w))

    # creating pad of zeros around the images
    padded_imgs = np.pad(images,
                         ((0, 0), (p_h, p_h), (p_w, p_w)),
                         mode="constant",
                         constant_values=0)

    # vectorizing the n_images into an array
    imgs_arr = np.arange(0, n_images)

    # iterating over the output array and generating the convolution
    for x in range(o_h):
        for y in range(o_w):
            x0 = x * s_h
            y0 = y * s_w
            x1 = x0 + k_h
            y1 = y0 + k_w
            outputs[imgs_arr, x, y] = np.sum(np.multiply(
                padded_imgs[imgs_arr, x0: x1, y0: y1], kernel), axis=(1, 2))

    return outputs
