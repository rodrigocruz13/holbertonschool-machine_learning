#!/usr/bin/env python3

"""
class TripletLoss
"""

import tensorflow as tf


class TripletLoss(tf.keras.layers.Layer):
    """
        Inherits from tensorflow.keras.layers.Layer
    """

    def __init__(self, alpha, **kwargs):
        """
        Constructor that sets the public instance attribute alpha

        Args:
            alpha:      α value used to calculate the triplet loss
        """

        self.alpha = alpha
        super(TripletLoss, self).__init__(**kwargs)

    def triplet_loss(self, inputs):
        """
        Function that calculate Triplet Loss:

        Args:
            - inputs:       list containing the anchor, positive and negative
                            output tensors from the last layer of the model
                            respectively
        Returns:
                        a tensor containing the triplet loss values
        """

        anc_output = inputs[0]
        pos_output = inputs[1]
        neg_output = inputs[2]

        d_posit = tf.reduce_sum(tf.square(anc_output - pos_output), 1)
        d_negat = tf.reduce_sum(tf.square(anc_output - neg_output), 1)

        margin = self.alpha
        loss = tf.maximum(margin + d_posit - d_negat, 0)

        return loss

    def call(self, inputs):
        """
        Args:
            - inputs:       list containing the anchor, positive, and negative
                            output tensors from the last layer of the model,
                            respectively adds the triplet loss to the graph
        Returns:
                            the triplet loss tensor
        """

        loss = self.triplet_loss(inputs)
        self.add_loss(loss)

        return loss
