#!/usr/bin/env python3

"""script for
"""

import tensorflow as tf


def change_brightness(image, max_delta):
    """[Function that randomly changes the brightness of an image]

    Args:
        image ([ 3D tf.Tensor]):    [tf.Tensor containing the image to flip]
        intensity                   maximum amount the image should be
                                    brightened (or darkened)
    Returns
        the sheared image
    """

    # TensorFlow. 'x' = A placeholder for an image.

    bright_img = tf.image.random_brightness(image, max_delta)

    return bright_img
