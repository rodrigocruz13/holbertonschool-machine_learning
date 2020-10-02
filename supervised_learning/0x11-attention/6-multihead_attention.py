#!/usr/bin/env python3
""" Module used to """

import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """[Class that inherits from tf.keras.layers.Layer 2 perform multihead att]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, dm, h):
        """[Class constructor]

        Args:
            dm ([int]): [dimensionality of the model. dm is divisible by h]
            h ([int]): [number of heads]

        Public instance attributes:
        - h:        the number of heads
        - dm:       the dimensionality of the model
        - depth:    the depth of each attention head
        - Wq:       Dense layer with dm units, used to create the query matrix
        - Wk:       Dense layer with dm units, used to create the key matrix
        - Wv:       Dense layer with dm units, used to create the value matrix
        - linear:   Dense layer with dm units, used to create the att output
        """

        # super() function that will make the child class inherit all the
        # methods and properties from its parent:

        super(MultiHeadAttention, self).__init__()
        self.dm = dm
        self.h = h
        self.depth = self.dm // self.h
        self.linear = tf.keras.layers.Dense(self.dm)
        self.Wq = tf.keras.layers.Dense(self.dm)
        self.Wk = tf.keras.layers.Dense(self.dm)
        self.Wv = tf.keras.layers.Dense(self.dm)

    def call(self, Q, K, V, mask):
        """[summary]

        Args:
        Q ([tensor]):       [is a tensor with its last two dimensions as
                            (..., seq_len_q, dk) containing the query matrix]

        K ([tensor]):       [is a tensor with its last two dimensions as
                            (..., seq_len_q, dk)  containing the key matrix]
        V ([tensor]):       [is a tensor with its last two dimensions as
                            (..., seq_len_q, dk) containing the value matrix]
        mask ([tensor], optional): [tensor that can be broadcast into
                                    (..., seq_len_q, seq_len_v) containing
                                    the optional mask]. Defaults to None.
        mask is always None

        Returns: output, weights
            output a tensor with its last two dimensions as
            (..., seq_len_q, dm) containing the scaled dot product attention
            weights a tensor with its last three dimensions as
            (..., h, seq_len_q, seq_len_v) containing the attention weights
        You should use

        """

        k = self.Wk(K)
        q = self.Wq(Q)
        v = self.Wv(V)

        batch_size = tf.shape(Q)[0]

        k = self.cut_heads(k, batch_size)
        q = self.cut_heads(q, batch_size)
        v = self.cut_heads(v, batch_size)

        scaled_att, weights = sdp_attention(q, k, v, mask)
        scaled_att = tf.transpose(scaled_att, perm=[0, 2, 1, 3])
        concat_att = tf.reshape(scaled_att, (batch_size, -1, self.dm))
        output = self.linear(concat_att)

        return output, weights

    def cut_heads(self, x, batch_size):
        """[summary]

        Args:
            x ([type]): [description]
            batch_size ([type]): [description]

        Returns:
            [type]: [description]
        """

        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
