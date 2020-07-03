#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs a valid convolution on grayscale images
    Args:
        - images is a numpy.ndarray with shape (m, h, w, c) containing
          multiple grayscale images
            - m is the number of examples
            - h_prev is the height of the previous layer
            - w_prev is the width of the previous layer
            - c_prev is the number of channels in the previous layer
        - kernel_shape is a numpy.ndarray with shape (kh, kw) containing
          the shape of the pooling
            - kh is the height of the kernel
            - kw is the width of the kernel
        - stride is a tuple of (sh, sw)
            - sh is the stride for the height of the image
            - sw is the stride for the width of the image
        - mode is a tuple of (sh, sw)
            - max indicates max pooling
            - avg indicates average pooling
    Returns:
        A numpy.ndarray containing the pooled images
    """

    _, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    x = A_prev

    if padding == 'valid':
        p_h = 0
        p_w = 0

    if padding == 'same':
        p_h = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        p_w = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    x_padded = np.pad(x, [(0, 0), (p_h, p_h), (p_w, p_w), (0, 0)],
                      mode='constant', constant_values=0)

    dW = np.zeros_like(W)
    dx = np.zeros(x_padded.shape)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for k in range(c_new):
                    dx[i,
                       h * (sh):(h * (sh)) + kh,
                       w * (sw):(w * (sw)) + kw,
                       :] += dZ[i, h, w, k] * W[:, :, :, k]

                    dW[:, :,
                       :, k] += x_padded[i,
                                         h * (sh):(h * (sh)) + kh,
                                         w * (sw):(w * (sw)) + kw,
                                         :] * dZ[i, h, w, k]
    if padding == 'same':
        dx = dx[:, p_h:-p_h, p_w:-p_w, :]
    else:
        dx = dx

    return dx, dW, db
