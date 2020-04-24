#!/usr/bin/env python3
"""
class Yolo that uses the Yolo v3 algorithm to perform object detection:
"""

import numpy as np
import tensorflow as tf
import glob
import cv2


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

    def sigmoid_f(self, x):
        """
            sigmoid.
        # Args
            x: Tensor.
        # Returns
            numpy ndarray.
        """

        return (1 / (1 + np.exp(-x)))

    def process_outputs(self, outputs, image_size):
        """
        Function that
        Args:
            - outputs:  list of numpy.ndarrays containing the predictions from
                        the Darknet model for a single image: Each output has
                        shape (grid_height, grid_width, anchor_boxes,
                        4 + 1 + classes)
                    > grid_height:  Height of the grid used for the output
                                    anchor_boxes
                    > grid_width:   Width of the grid used for the output
                                    anchor_boxes
                    > anchor_boxes: Number of anchor boxes used
                    > 4:
                        t_x:    x pos of the center point of the anchor box
                        t_y:    y pos of the center point of the anchor box
                        t_w:    width of the anchor box
                        t_h:    height of the anchor box
                    > 1:            box_confidence
                    > classes:      class probabilities for all classes

            - image_size:   numpy.ndarray containing the image’s original size
                            [image_height, image_width]

        Returns:
            A tuple of (boxes, box_confidences, box_class_probs):
                    > boxes:    List of numpy.ndarrays of shape (grid_height,
                                grid_width, anchor_boxes, 4) containing the
                                processed boundary boxes for each output,
                                respectively:
                            4:  (x1, y1, x2, y2) should represent the boundary
                                box relative to original image
                    > box_confidences:
                                list of numpy.ndarrays of shape (grid_height,
                                grid_width, anchor_boxes, 1) containing the box
                                confidences for each output, respectively
                    > box_class_probs:
                                list of numpy.ndarrays of shape (grid_height,
                                grid_width, anchor_boxes, classes) containing
                                the box’s class probabilities for each output,
                                respectively
        """

        # shape (13,  13,   3,  [t_x, t_y, t_w, t_h],   1    80)
        # Dim   ([0], [1], [2],        [3],           [4]   [5])

        # 1. boxes = dim [2]
        # Procesed according to Fig 2 of paper: https://bit.ly/3emqWp0
        # Adapted from https://bit.ly/2VEZgmZ

        boxes = []
        for i in range(len(outputs)):
            boxes_i = outputs[i][..., 0:4]
            grid_h_i = outputs[i].shape[0]
            grid_w_i = outputs[i].shape[1]
            anchor_box_i = outputs[i].shape[2]

            for anchor_n in range(anchor_box_i):
                for cy_n in range(grid_h_i):
                    for cx_n in range(grid_w_i):

                        tx_n = outputs[i][cy_n, cx_n, anchor_n, 0:1]
                        ty_n = outputs[i][cy_n, cx_n, anchor_n, 1:2]
                        tw_n = outputs[i][cy_n, cx_n, anchor_n, 2:3]
                        th_n = outputs[i][cy_n, cx_n, anchor_n, 3:4]

                        # size of the anchors
                        pw_n = self.anchors[i][anchor_n][0]
                        ph_n = self.anchors[i][anchor_n][1]

                        # calculating center
                        bx_n = self.sigmoid_f(tx_n) + cx_n
                        by_n = self.sigmoid_f(ty_n) + cy_n

                        # calculating hight and width
                        bw_n = pw_n * np.exp(tw_n)
                        bh_n = ph_n * np.exp(th_n)

                        # generating new center
                        new_bx_n = bx_n / grid_w_i
                        new_by_n = by_n / grid_h_i

                        # generating new hight and width
                        new_bh_n = bh_n / self.model.input.shape[2].value
                        new_bw_n = bw_n / self.model.input.shape[1].value

                        # calculating (cx1, cy1) and (cx2, cy2) coords
                        y1 = (new_by_n - (new_bh_n / 2)) * image_size[0]
                        y2 = (new_by_n + (new_bh_n / 2)) * image_size[0]
                        x1 = (new_bx_n - (new_bw_n / 2)) * image_size[1]
                        x2 = (new_bx_n + (new_bw_n / 2)) * image_size[1]

                        boxes_i[cy_n, cx_n, anchor_n, 0] = x1
                        boxes_i[cy_n, cx_n, anchor_n, 1] = y1
                        boxes_i[cy_n, cx_n, anchor_n, 2] = x2
                        boxes_i[cy_n, cx_n, anchor_n, 3] = y2

            boxes.append(boxes_i)

        # 2. box confidence = dim [4]
        confidence = []
        for i in range(len(outputs)):
            confidence_i = self.sigmoid_f(outputs[i][..., 4:5])
            confidence.append(confidence_i)

        # 3. box class_probs = dim [5:]
        probs = []
        for i in range(len(outputs)):
            probs_i = self.sigmoid_f(outputs[i][:, :, :, 5:])
            probs.append(probs_i)

        return (boxes, confidence, probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
        Function that
        Args:
            - boxes:            List of numpy.ndarrays of shape (grid_height,
                                grid_width, anchor_boxes, 4) containing the
                                processed boundary boxes for each output,
                                respectively
            - box_confidences:  list of numpy.ndarrays of shape (grid_height,
                                grid_width, anchor_boxes, 1) containing the
                                processed box confidences for each output,
                                respectively
            - box_class_probs:  list of numpy.ndarrays of shape (grid_height,
                                grid_width, anchor_boxes, classes) containing
                                the processed box class probabilities for each
                                output, respectively

        Returns:
            A tuple of (filtered_boxes, box_classes, box_scores):
                > filtered_boxes:   numpy.ndarray of shape (?, 4) containing
                                    all of the filtered bounding boxes
                > box_classes:      a numpy.ndarray of shape (?,) containing
                                    the class number that each box in
                                    filtered_boxes predicts, respectively
                > box_scores:       numpy.ndarray of shape (?) containing the
                                    box scores for each box in filtered_boxes,
                                    respectively
        """

        scores = []
        classes = []
        box_classes_scores = []
        index_arg_max = []
        box_classes = []

        # 1. Multiply confidence x probs to find real confidence of each class
        for bc_i, probs_j in zip(box_confidences, box_class_probs):
            scores.append(bc_i * probs_j)

        # 2. find temporal indices de clas cajas con los arg mas altos
        for box_scoreee in scores:
            index_arg_max = np.argmax(box_scoreee, axis=-1)
            # -1 = last dimension)

            # 3. Flatten each array
            index_arg_max_flat = index_arg_max.flatten()

            # 4. Everything in one single array
            classes.append(index_arg_max_flat)

            # find the values
            score_max = np.max(box_scoreee, axis=-1)
            score_max_flat = score_max.flatten()
            box_classes_scores.append(score_max_flat)

        boxes = [box.reshape(-1, 4) for box in boxes]
        # (13, 13, 3, 4) ----> (507, 4)

        box_classes = np.concatenate(classes, axis=-1)
        # -1 = add to the end

        box_classes_scores = np.concatenate(box_classes_scores, axis=-1)
        # -1 = add to the end

        boxes = np.concatenate(boxes, axis=0)

        # filtro
        # boxes[box_classes_scores >= self.class_t]
        filtro = np.where(box_classes_scores >= self.class_t)

        return (boxes[filtro], box_classes[filtro], box_classes_scores[filtro])

    def iou(self, x1, x2, y1, y2, pos1, pos2, area):
        """
        Function that
        Args:
            - x1:       xxx
            - x2:       xxx
            - y1        xxx
            - yy2       xxx
            - pos1      xxx
            - pos2      xxx
            - area      xxx

        Returns:
            The intersection over union %

        """

        # find the coordinates
        a = np.maximum(x1[pos1], x1[pos2])
        b = np.maximum(y1[pos1], y1[pos2])

        c = np.minimum(x2[pos1], x2[pos2])
        d = np.minimum(y2[pos1], y2[pos2])

        height = np.maximum(0.0, d - b)
        width = np.maximum(0.0, c - a)

        # overlap ratio betw bounding box
        intersection = (width * height)
        union = area[pos1] + area[pos2] - intersection
        iou = intersection / union

        return iou

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
        Args:
            - filtered_boxes:   numpy.ndarray of shape (?, 4) containing all
                                of the filtered bounding boxes
            - box_classes:      numpy.ndarray of shape (?,) containing the
                                class number 4 the class that filtered_boxes
                                predicts, respectively
            - box_scores:       numpy.ndarray of shape (?) containing the box
                                scores for each box in filtered_boxes,
                                respectively
        Returns:                Tuple of (box_predictions,
                                          predicted_box_classes,
                                          predicted_box_scores):
                > box_predictions:          numpy.ndarray of shape (?, 4)
                                            containing all of the predicted
                                            bounding boxes ordered by class &
                                            box score
                > predicted_box_classes:    numpy.ndarray of shape (?,)
                                            containing the class number for
                                            box_predictions ordered by class &
                                            box score, respectively
                > predicted_box_scores:     numpy.ndarray of shape (?)
                                            containing the box scores for
                                            box_predictions ordered by class &
                                            box score, respectively
        """

        box_predictions = []
        predicted_box_classes = []
        predicted_box_scores = []

        for classes in set(box_classes):
            index = np.where(box_classes == classes)

            # function arrays
            filtered = filtered_boxes[index]
            scores = box_scores[index]
            classe = box_classes[index]

            # coordinates of the bounding boxes
            x1 = filtered[:, 0]
            y1 = filtered[:, 1]
            x2 = filtered[:, 2]
            y2 = filtered[:, 3]

            # calculate area of the bounding boxes and sort from high to low
            area = (x2 - x1) * (y2 - y1)
            index_list = np.flip(scores.argsort(), axis=0)

            # loop remaining indexes to hold list of picked indexes
            keep = []
            while (len(index_list) > 0):
                pos1 = index_list[0]
                pos2 = index_list[1:]
                keep.append(pos1)

                # find the intersection over union %
                iou = self.iou(x1, x2, y1, y2, pos1, pos2, area)

                below_threshold = np.where(iou <= self.nms_t)[0]
                index_list = index_list[below_threshold + 1]

            # array of piked indexes
            keep = np.array(keep)

            box_predictions.append(filtered[keep])
            predicted_box_classes.append(classe[keep])
            predicted_box_scores.append(scores[keep])

        box_predictions = np.concatenate(box_predictions)
        predicted_box_classes = np.concatenate(predicted_box_classes)
        predicted_box_scores = np.concatenate(predicted_box_scores)

        return (box_predictions, predicted_box_classes, predicted_box_scores)

    @staticmethod
    def load_images(folder_path):
        """
        Args:
            - folder_path:  a string representing the path to the folder
                            holding all the images to load
        Returns
            - tuple of (images, image_paths):
                > images:       List of images as numpy.ndarrays
                > image_paths:  List of paths of each image in images
        """

        # creating a correct full path argument
        images = []
        image_paths = glob.glob(folder_path + '/*', recursive=False)

        # creating the images list
        for imagepath_i in image_paths:
            images.append(cv2.imread(imagepath_i))

        return(images, image_paths)
