#!/usr/bin/env python3
""" Module used to """


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ class RNNEncoder
        class that inherits from tensorflow.keras.layers.Layer to calculate
        the attention for machine translation based on this paper:
        https://arxiv.org/pdf/1409.0473.pdf

    Args:
        tf ([type]): [description]
    """

    def __init__(self, units):
        """[Class constructor]

        Args:
            units       ([int]):    [number of hidden units in the RNN cell]

        Sets the following public instance attributes:
            W:      A Dense layer with units units, to be applied to the
                    previous decoder hidden state
            U:      A Dense layer with units units, to be applied to the
                    encoder hidden states
            V:      A Dense layer with 1 units, to be applied to the tanh
                    of the sum of the outputs of W and U
        """

        # super() function that will make the child class inherit all the
        # methods and properties from its parent:

        super(SelfAttention, self).__init__()

        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)
        self.W = tf.keras.layers.Dense(units)

    def call(self, s_prev, hidden_states):
        """[Public instance method]

        Args:
            s_prev ([tensor]):  [s a tensor of shape (batch, units) containing
                                the previous decoder hidden state]
            hidden_states ([type]): [description]

        Returns:
            [type]: [description]
        """
        query = tf.expand_dims(s_prev, 1)

        # V - a Dense layer with 1 units, to be applied to the tanh of the
        # sum of the outputs of W and U

        tfadd = tf.math.add(self.W(query), self.U(hidden_states))
        score = self.V(tf.nn.tanh(tfadd))
        weigh = tf.nn.softmax(score, axis=1)
        hs_we = weigh * hidden_states
        context = tf.reduce_sum(hs_we, axis=1)

        return context, weigh
