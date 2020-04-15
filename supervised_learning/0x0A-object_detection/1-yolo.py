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

        num_outpust = len(outputs) # num = 3
        #print(outputs.__dict__)

        for i in range(len(outputs)):
            # grid_height, grid_width, anchor_boxes
            x = outputs[i]
            #print (i, type(x))
            # len (output[0]) =  13
            # len (output[1]) =  26
            # len (output[2]) =  52
            #print("\t")
            for j in range(len(x)):
                #print("\t", i, j, type(x[j]))
                y = x[j]
                for k in range(len(y)):
                    #print("\t\t", i, j, k, type(y[k]))
                    z = y[k]
                    for l in range(len(z)):
                        za = z[l]
                        for m in range (len(za)): 
                            pass
            print("\t",i, "\t",j, "\t",k, "\t",l, "\t",m)
            #print(outputs[0][12][12][2])
            #print(outputs[0][12][12][2][84])
            #print(outputs[0][: , : , : , 3:5])

            #print(type(x), x)
            #print(type(y), y)
            #print(type(z), z)
        #print(type(za))
        #print(za)
            
        print("--------------------")
            
        """
            print("grid-height = ", prediction[0])
            print("grid-width = ", prediction[1])
            print("anchor_boxes = ", prediction[2])
            print("t_x = ", prediction[3][0])
            print("t_y = ", prediction[3][1])
            print("t_w = ", prediction[3][2])
            print("t_h = ", prediction[3][3])
            print("box_confidence = ", prediction[4])
        """
        print(image_size)
        box_confidences = [] 
        for i in range(len(outputs)):
            box_confidences.append(tf.math.sigmoid(outputs[i][:, :, :, 4:5]))
        
        print("outputs = ", len(outputs), type(outputs) )
        print("outputs[i] = ", len(outputs[i]), outputs[i].shape, type(outputs[i]))
        
        print("outputs[0] = ", len(outputs[0]), outputs[0].shape, type(outputs[0]))
        print("outputs[1] = ", len(outputs[1]), outputs[1].shape, type(outputs[1]))
        print("outputs[2] = ", len(outputs[2]), outputs[2].shape, type(outputs[2]))


        print("outputs[i][j] = ", len(outputs[i][j]), outputs[i][j].shape, type(outputs[i][j]))
        print("outputs[i][j][k] = ", len(outputs[i][j][k]), outputs[i][j][k].shape, type(outputs[i][j][k]))
        print("outputs[i][j][k][l] = ", len(outputs[i][j][k][l]), outputs[i][j][k][l].shape, type(outputs[i][j][k][l]))
#        print("outputs[i][j][k][l][m] = ", len(outputs[i][j][k][l][m]), outputs[i][j][k][l][m].shape, type(outputs[i][j][k][l][m]))

        return([], [], [])
