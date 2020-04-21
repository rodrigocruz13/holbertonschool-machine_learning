#!/usr/bin/env python3
"""
class Yolo that uses the Yolo v3 algorithm to perform object detection:
"""

import numpy as np
import cv2
import glob


def load_images(images_path, as_array=True):
    """
    Function that loads images from a directory or file:

    Args:
        - images_path:  Path to a directory from which to load images
        - as_array:     Boolean indicating whether the images should be
                        loaded as one numpy.ndarray
                        - If True, the images should be loaded as a
                          numpy.ndarray of shape (m, h, w, c) where:
                            > m is the number of images
                            > h is the height of the image
                            > w,is the width of the image
                            > c is the num of channels of all images
                        - If False, the images should be loaded as a list
                          of individual numpy.ndarrays

    Returns
        images, filenames
        - images:       Is either a list/numpy.ndarray of all images
        - filenames     is a list of the filenames associated with each image
                        in images
    """

    # creating a correct full path argument
    images_locations = glob.glob(images_path + '/*', recursive=False)
    filenames = sorted([s.replace('HBTN/', '') for s in images_locations])

    # creating the images lists
    images_files = []

    for file_i in filenames:
        long_path = 'HBTN/' + file_i
        image_i = cv2.imread(long_path)
        if (as_array):
            m = 1
            h, w, c = image_i.shape
            images_files.append(np.array([m, h, w, c]))
        else:
            images_files.append(image_i)

    return images_files, filenames
