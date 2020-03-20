#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def convolve_grayscale_same(images, kernel):
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

    # pad_h ⊛ = int (k_h - 1)/2
    # pad_w ⊛ = int (k_w - 1)/2
    p_h = int((k_h - 1) / 2)
    p_w = int((k_w - 1) / 2)

    if k_h % 2 == 0:
        p_h = int(k_h / 2)

    if k_w % 2 == 0:
        p_w = int(k_w / 2)

    # output_height and output_width
    # H = i_h + 2pad - k_h + 1, W = i_w + 2pad - k_w + 1
    o_h = i_h + 2 * p_h - k_h + 1
    o_w = i_w + 2 * p_w - k_w + 1

    if k_h % 2 == 0:
        o_h = i_h + 2 * p_h - k_h

    if k_w % 2 == 0:
        o_w = i_w + 2 * p_w - k_w

    # creating outputs of size: n_images, o_h x o_w
    outputs = np.zeros((n_images, o_h, o_w))

    # creating pad of zeros around the output images
    padded_imgs = np.pad(images,
                         pad_width=((0, 0), (p_h, p_h), (p_w, p_w)),
                         mode="constant",
                         constant_values=0)

    # vectorizing the n_images into an array
    imgs_arr = np.arange(0, n_images)

    # iterating over the output array and generating the convolution
    for x in range(o_h):
        for y in range(o_w):
            x1 = x + k_h
            y1 = y + k_w
            outputs[imgs_arr, x, y] = np.sum(np.multiply(
                padded_imgs[imgs_arr, x: x1, y: y1], kernel), axis=(1, 2))

    return outputs
