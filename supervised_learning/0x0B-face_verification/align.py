#!/usr/bin/env python3
"""
class FaceAlign
"""

import dlib
import numpy as np
import cv2
import os


class FaceAlign:
    """ Class constructor """

    def __init__(self, shape_predictor_path):
        """
        Class initializator
        Args:
            - shape_predictor_path:     path to the dlib shape predictor model

        Public instance attributes:
            - detector:                 contains dlib‘s default face detector
            - shape_predictor:          contains the dlib.shape_predictor
        """

        route1 = shape_predictor_path
        route2 = 'models/shape_predictor_68_face_landmarks.dat'
        route = route1 if os.path.exists(route1) else route2

        self.detector = dlib.get_frontal_face_detector()
        self.shape_predictor = dlib.shape_predictor(route)

    def detect(self, image):
        """
        Function that detects a face in an image:
        Args:
            - image:    numpy.ndarray of rank 3 containing an image from which
                        to detect a face

        Returns:        a dlib.rectangle containing the boundary box for the
                        face in the image, or None on failure
                - If multiple faces are detected, return the dlib.rectangle
                  with the largest area
                - If no faces are detected, return a dlib.rectangle that is
                 the same as the image
        """

        # find faces, on error return None
        # The 1 in the second argument indicates that we should upsample
        # the image 1 time. This will make everything bigger and allow us
        # to detect more faces.:
        try:
            faces_in_pic = self.detector(image, 1)
        except BaseException:
            return None

        # 0 faces detected: return dlib.rectangle(image)
        if len(faces_in_pic) == 0:
            return dlib.rectangle(0, 0, image.shape[1], image.shape[0])

        # 1 or more face detected: return max_area(face_i)
        max_ = 0
        max_face_i = 0
        for rectangle_i in faces_in_pic:
            if rectangle_i.area() > max_:
                max_ = rectangle_i.area()
                max_face_i = rectangle_i

        return max_face_i

    def find_landmarks(self, image, detection):
        """
        Function that finds facial landmarks:
        Args:
            - image:        numpy.ndarray of an image from which to find
                            facial landmarks
            - detection:    dlib.rectangle containing the boundary box of the
                            face in the image

        Returns:            numpy.ndarray of shape (p, 2) containing the
                            landmark points, or None on failure
                                > p     number of landmark points
                                > 2     x and y coordinates of the point

        """
        # Taken from:
        # www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python

        # initialize the list of (x, y)-coordinates
        coords = np.zeros((68, 2), dtype='int')

        # find landmarks
        shape = self.shape_predictor(image, detection)

        # loop over the 68 facial landmarks and convert them
        # to a 2-tuple of (x, y)-coordinates

        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)

        # return the list of (x, y)-coordinates
        return np.asarray(coords)

    def align(self, image, landmark_indices, anchor_points, size=96):
        """
        Function that aligns an image for face verification:
        Args:
            - image:            numpy.ndarray of rank 3 containing the image
                                to be aligned
            - landmark_indices  numpy.ndarray of shape (3,) containing the
                                indices of the three landmark points that
                                should be used for the affine transformation
            - anchor_points     numpy.ndarray of shape (3, 2) containing the
                                destination points for the affine
                                transformation, scaled to the range [0, 1]
            - size              desired size of the aligned image

        Returns:
            -   numpy.ndarray of shape (size, size, 3) containing the aligned
                image, or None if no face is detected
        """

        # find faces, on error return None:
        try:
            rectangles_in_pic = self.detect(image)
        except BaseException:
            return None

        # Detected at least 1 face, then extract landmarks
        landmarks_in_pic = self.find_landmarks(image, rectangles_in_pic)

        # Select eyes & nose as landmarks
        eyes_n_nose = landmarks_in_pic[landmark_indices].astype('float32')

        # Generate the transformation to the desired size(anchor_points * size)
        tr = cv2.getAffineTransform(eyes_n_nose, anchor_points * size)

        # transform the source image by the transformation 'tr' already done
        new_image = cv2.warpAffine(image, tr, (size, size))

        return new_image
