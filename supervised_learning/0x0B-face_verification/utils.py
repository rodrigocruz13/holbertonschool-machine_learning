#!/usr/bin/env python3
"""
class Yolo that uses the Yolo v3 algorithm to perform object detection:
"""

import numpy as np
import cv2
import glob
import os


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
    images_locations = glob.glob(images_path + '/*', recursive=True)
    filenames = sorted([s.replace(images_path + '/', '')
                        for s in images_locations])

    # creating the images lists
    images_files = []

    for file_i in filenames:
        long_path = images_path + '/' + file_i
        image_i = cv2.imread(long_path)

        # making sure images are RGB formatted
        if (len(image_i.shape) == 3):

            b, g, r = cv2.split(image_i)
            image_i = cv2.merge([r, g, b])

        images_files.append(image_i)

    if (not as_array):
        return images_files, filenames

    return np.asarray(images_files), filenames


def load_csv(csv_path, params={}):
    """
    Function that loads the contents of a csv file as a list of lists:

    Args:
        - csv_path:     Path to the csv to load
        - params:       Parameters to load the csv with

    Returns
        - []            List of lists representing the contents
                        found in csv_path
    """

    import csv

    args = ','.join(['%s=%s' % x for x in params.items()])
    if (len(args) > 0):
        file_Reader = csv.reader(open(csv_path, args))
    else:
        file_Reader = csv.reader(open(csv_path))

    csv_info = []

    for row in file_Reader:
        csv_info.append(row)
    return csv_info


def save_images(path, images, filenames):
    """
    Function that saves images to a specific path:

    Args:
    - path:         Path to the directory in which the images should be saved
    - images:       list/numpy.ndarray of images to save
    - filenames:    list of filenames of the images to save

    Returns:        True on success and False on failure

    """

    if not (os.path.exists(path)):
        return False

    i = 0
    for image_i in images:
        b, g, r = cv2.split(image_i)
        image_i = cv2.merge([r, g, b])
        cv2.imwrite("./{}/{}".format(path, filenames[i]), image_i)
        i = i + 1
    return True


def generate_triplets(images, filenames, triplet_names):
    """ generates the triplets
    Arg:
        images: np.ndarray shape (n, h, w, 3) with images in the dataset
        filenames: list leng n with the corresponding filenames for images
        triplet_names: list of lists where each sublist contains the
                 filenames of an anchor, positive, and negative image
    Returns: a list [A, P, N]
       - A np.ndarray shape (m, h, w, 3) has anchor images for all m triplets
       - P np.ndarray (m, h, w, 3) has the positive images for all m triplets
       - N np.ndarray (m, h, w, 3) has the negative images for all m triplets
    """

    # full list of names
    all_ = [filenames[i].split('.')[0] for i in range(len(filenames))]

    # list of names of anchor (A), positive (P) and negative (N) images
    anc = [x for row in triplet_names for x in row[0].split('.')]
    pos = [x for row in triplet_names for x in row[1].split('.')]
    neg = [x for row in triplet_names for x in row[2].split('.')]

    # all_list indexes of anchor (A), positive (P) and negative (N) images
    a_i = [all_.index(anc[x]) for x in range(len(anc)) if anc[x] in all_]
    p_i = [all_.index(pos[x]) for x in range(len(pos)) if pos[x] in all_]
    n_i = [all_.index(neg[x]) for x in range(len(neg)) if neg[x] in all_]

    return [images[a_i], images[p_i], images[n_i]]
