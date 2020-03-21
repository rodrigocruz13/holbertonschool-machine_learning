#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
    Performs a valid convolution on images with multiple channels (colors)
    Args:
        - images is a numpy.ndarray with shape (m, h, w, c) containing
          multiple grayscale images
            - m is the number of images
            - h is the height in pixels of the images
            - w is the width in pixels of the images
            - c is the number of channels in the image
        - kernels is a numpy.ndarray with shape (kh, kw, c)
          containing the kernel for the convolution
            - kh is the height of the kernel
            - kw is the width of the kernel
            - nc is the number of kernels
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

    # images channel
    i_c = images.shape[3]

    # kernel_width and kernel_height
    k_h = kernels.shape[0]
    k_w = kernels.shape[1]

    # numer of channels of the kernel
    k_c = kernels.shape[3]

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
    o_h = int((i_h + 2 * p_h - k_h) / s_h) + 1
    o_w = int((i_w + 2 * p_w - k_w) / s_w) + 1

    # creating outputs of size: [n_images,  o_h  ⊛  o_w  ⊛  k_c ⊛  i_c]
    outputs = np.zeros((n_images, o_h, o_w, k_c))

    # creating pad of zeros around the output images
    padded_imgs = np.pad(images,
                         ((0, 0),       # dim n_images
                          (p_h, p_h),   # dim height
                          (p_w, p_w),   # dim width
                          (0, 0)        # dim channels
                          ),
                         mode="constant",
                         constant_values=0)

    # vectorizing the n_images into an array (creating a new dimension)
    imgs_arr = np.arange(0, n_images)

    # iterating over the output array and generating the convolution
    for x in range(o_h):
        for y in range(o_w):
            for z in range(k_c):
                x0 = x * s_h
                y0 = y * s_w
                x1 = x0 + k_h
                y1 = y0 + k_w
                outputs[imgs_arr, x, y, z] = np.sum(np.multiply(
                    padded_imgs[imgs_arr, x0: x1, y0: y1],
                    kernels[:, :, :, z]), axis=(1, 2, 3))

    return outputs
