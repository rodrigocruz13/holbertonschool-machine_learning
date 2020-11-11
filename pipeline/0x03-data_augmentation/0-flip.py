#!/usr/bin/env python3

"""script for
"""

import tensorflow as tf


def flip_image(image):
    """[Function that flips an image horizontally]

    Args:
        image ([ 3D tf.Tensor]): [tf.Tensor containing the image to flip]

    Returns
        the flipped image
    """

    # TensorFlow. 'x' = A placeholder for an image.

    flipped_image = tf.image.flip_left_right(image)

    return flipped_image
