#!/usr/bin/env python3

"""script for
"""

import tensorflow as tf


def rotate_image(image):
    """[Function that flips an image horizontally]

    Args:
        image ([ 3D tf.Tensor]): [tf.Tensor containing the image to flip]

    Returns
        the flipped image
    """

    # TensorFlow. 'x' = A placeholder for an image.

    rotated_image = tf.image.rot90(image)

    return rotated_image
