#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
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

    # output_height and output_width
    o_h = i_h - k_h + 1
    o_w = i_w - k_w + 1

    # pad ⊛
    pad = 1

    # creating outputs of size: n_images, o_h x o_w
    outputs = np.zeros((n_images, o_h, o_w))

    # vectorizing the n_images
    images_array = np.arange(0, n_images)

    # iterating over the output array and generating the convolution
    for x in range(o_h):
        for y in range(o_w):
            x1 = x + k_h
            y1 = y + k_w
            outputs[images_array, x, y] = np.sum(np.multiply(
                images[images_array, x: x1, y: y1], kernel), axis=(1, 2))

    return outputs
