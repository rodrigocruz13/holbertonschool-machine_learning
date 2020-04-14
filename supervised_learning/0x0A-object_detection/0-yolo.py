#!/usr/bin/env python3
"""
class Yolo that uses the Yolo v3 algorithm to perform object detection:
"""

import numpy as np
import tensorflow as tf


class Yolo:
    """ Class constructor """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Class initializator
        Args:
            - model_path:   Path to where a Darknet Keras model is stored
            - classes_path: Path to where the list of class names used for the
                            Darknet model, listed by index order, can be found
            - class_t:      Float representing the box score threshold for the
                            initial filtering step
            - nms_t:        Float representing the IOU threshold for non-max
                            suppression
            - anchor:       Numpy.ndarray of shape (outputs, anchor_boxes, 2)
                            containing all of the anchor boxes:
                    > outputs:      Number of outputs (predictions) made by
                                    the Darknet model
                    > anchor_boxes: Number of anchor boxes used for each
                                    prediction
                    > 2:            [anchor_box_width, anchor_box_height]

        Public instance attributes:
            - model:        the Darknet Keras model
            - class_names:  list of the class names for the model
            - class_t:      the box score threshold for the initial
                            filtering step
            - nms_t:        the IOU threshold for non-max suppression
            - anchors:      the anchor boxes
        """

        # 1. Load the h5 file witht the description of the NN model
        model = tf.keras.models.load_model(model_path)

        # 2. Load the classes names
        classes = open(classes_path).read()

        # 3. Save into public instance attributes
        self.model = model
        self.class_names = classes.splitlines()
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
