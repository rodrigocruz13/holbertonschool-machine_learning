#!/usr/bin/env python3

"""script for
"""

import numpy as np
import tensorflow as tf


def pca_color(image, alphas):
    """[Function that randomly changes the brightness of an image]

    Args:
        image ([ 3D tf.Tensor]):    [tf.Tensor containing the image to flip]
        alphas                      a tuple of length 3 containing the amount
                                    that each channel should change
    Returns
        the augmented image
    """

    # convert image into array (Converts a 3D nparray to a PIL Image instance)
    img_array = tf.keras.preprocessing.image.img_to_array(image)

    # source https://github.com/
    # pixelatedbrian/fortnight-furniture/blob/master/src/fancy_pca.py

    # generate copy as float values
    orig_img = img_array.astype(float).copy()
    img = img_array.astype(float).copy()

    # normalize
    img = img / 255.0

    # flatten image to columns of RGB
    img_rs = img.reshape(-1, 3)
    # img_rs shape (640000, 3)

    # center mean
    img_centered = img_rs - np.mean(img_rs, axis=0)

    # paper says 3x3 covariance matrix
    img_cov = np.cov(img_centered, rowvar=False)

    # eigen values and eigen vectors
    eig_vals, eig_vecs = np.linalg.eigh(img_cov)

    # sort values and vector
    sort_perm = eig_vals[::-1].argsort()
    eig_vals[::-1].sort()
    eig_vecs = eig_vecs[:, sort_perm]

    # get [p1, p2, p3]
    m1 = np.column_stack((eig_vecs))

    # get 3x1 matrix of eigen values multiplied by random variable draw from
    # normal distribution with mean of 0 and standard deviation of 0.1
    m2 = np.zeros((3, 1))
    # according to the paper alpha should only be draw once per augmentation
    # (not once per channel)
    # alpha = np.random.normal(0, alphas). It is already given

    # broad cast to speed things up
    m2[:, 0] = alphas * eig_vals[:]

    # this is the vector that we're going to add to each pixel in a moment
    add_vect = np.matrix(m1) * np.matrix(m2)

    for idx in range(3):   # RGB
        orig_img[..., idx] += add_vect[idx]

    # for image processing it was found that working with float 0.0 to 1.0
    # was easier than integers between 0-255
    # orig_img /= 255.0
    orig_img = np.clip(orig_img, 0.0, 255.0)

    # orig_img *= 255
    orig_img = orig_img.astype(np.uint8)

    # about 100x faster after vectorizing the numpy, it will be even faster
    # later since currently it's working on full size images and not small,
    # square images that will be fed in later as part of the post processing
    # before being sent into the model
    # print("elapsed time: {:2.2f}".format(time.time() - start_time), "\n")

    return orig_img
