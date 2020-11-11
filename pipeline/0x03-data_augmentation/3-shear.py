#!/usr/bin/env python3

"""script for
"""

import tensorflow as tf


def shear_image(image, intensity):
    """[Function that flips an image horizontally]

    Args:
        image ([ 3D tf.Tensor]):    [tf.Tensor containing the image to flip]
        intensity                   intensity with which the image will be
                                    sheared
    Returns
        the sheared image
    """

    # convert image into array (Converts a 3D nparray to a PIL Image instance)
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    sheared_img = tf.keras.preprocessing.image.random_shear(img_array,
                                                            intensity,
                                                            row_axis=0,
                                                            col_axis=1,
                                                            channel_axis=2)
    # convert back array into image (Converts a PIL Img instance to a nparray)
    sheared_img = tf.keras.preprocessing.image.array_to_img(sheared_img)

    return sheared_img
