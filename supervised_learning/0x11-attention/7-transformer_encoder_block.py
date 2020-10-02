#!/usr/bin/env python3
""" Module used to """

import tensorflow as tf
MultiHeadAttention = __import__('6-multi_head_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Transformer Encoder Block
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """[summary]

        Args:
            dm ([type]): [the dimensionality of the model]
            h ([type]): [the number of heads]
            hidden ([type]): [hidden units in the fully connected layer]
            drop_rate (float, optional): [the dropout rate]. Defaults to 0.1.


        Public instance attributes:
        - mha:              a MultiHeadAttention layer
        - dense_hidden:     the hidden dense layer with hidden units and relu
                            activation
        - dense_output:     the output dense layer with dm units
        - layernorm1:       the first layer norm layer, with epsilon=1e-6
        - layernorm2:       the second layer norm layer, with epsilon=1e-6
        - dropout1:         the first dropout layer
        - dropout2:         the second dropout layer
        """

        # super() function that will make the child class inherit all the
        # methods and properties from its parent:

        super(EncoderBlock, self).__init__()
        activ = 'relu'
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation=activ)
        self.dense_output = tf.keras.layers.Dense(dm)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)


    def call(self, x, training, mask=None):
        """[summary]

        Args:
            x ([type]): [tensor of shape (batch, input_seq_len, dm)containing
                        the input to the encoder block]
            training ([type]): [a boolean to find if the model is training]
            mask ([type], optional): [the mask to be applied for multi head
                                     attention]. Defaults to None.

        Returns:
            [type]: [a tensor of shape (batch, input_seq_len, dm) containing
            the block’s output]
        """

        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)

        n = x + attn_output
        out1 = self.layernorm1(n)
        output = self.dense_hidden(out1)
        output = self.dense_output(output)
        output = self.dropout2(output, training=training)

        m = out1 + output
        out = self.layernorm2(m)

        return out

