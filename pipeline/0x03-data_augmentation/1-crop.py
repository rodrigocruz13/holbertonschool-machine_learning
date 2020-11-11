#!/usr/bin/env python3

"""script for
"""

import tensorflow as tf


def crop_image(image, size):
    """[Function that flips an image horizontally]

    Args:
        image ([ 3D tf.Tensor]): [tf.Tensor containing the image to flip]
        size is a tuple containing the size of the crop
    Returns
        the flipped image
    """

    # TensorFlow. 'x' = A placeholder for an image.

    cropped_image = tf.image.random_crop(image, size)

    return cropped_image
