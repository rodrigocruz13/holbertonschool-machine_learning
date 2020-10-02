#!/usr/bin/env python3
""" Module used to """

import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Transformer Decoder Block
    """

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """[ Class constructor]

        Args:
            dm ([type]): [the dimensionality of the model]
            h ([type]): [the number of heads]
            hidden ([type]): [# of hidden units in the fully connected layer]
            drop_rate (float, optional): [dropout rate]. Defaults to 0.1.

        Public instance attributes:
        - mha1:         the first MultiHeadAttention layer
        - mha2:         the second MultiHeadAttention layer
        - dense_hidden: the hidden dense layer with hidden units and relu
                        activation
        - dense_output: the output dense layer with dm units
        - layernorm1:   the first layer norm layer, with epsilon=1e-6
        - layernorm2:   the second layer norm layer, with epsilon=1e-6
        - layernorm3:   the third layer norm layer, with epsilon=1e-6
        - dropout1:     the first dropout layer
        - dropout2:     the second dropout layer
        - dropout3:     the third dropout layer
        """

        # super() function that will make the child class inherit all the
        # methods and properties from its parent:

        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """[Public instance method]

        Args:
            x ([type]): [a tensor of shape (batch, target_seq_len, dm) with
                         the input to the decoder block]
            encoder_output ([type]): [a tensor of shape (batch, input_seq_len,
                                     dm) with the output of the encoder]
            training ([type]): [a boolean 2 find if the model is training]
            look_ahead_mask ([type]): [the mask to be applied to the first
                                       multi head attention layer]
            padding_mask ([type]): [the mask 2 b applied to the 2nd multi head
                                    attention layer]

        Returns:
            [type]: [a tensor of shape (batch, target_seq_len, dm) containing
                     the block’s output]
        """

        # (batch_size, target_seq_len, d_model)
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        n = attn1 + x
        out1 = self.layernorm1(n)

        attn2, attn_weights_block2 = self.mha2(out1,
                                               encoder_output,
                                               encoder_output,
                                               padding_mask)
        attn2 = self.dropout2(attn2, training=training)

        m = attn2 + out1
        y = self.layernorm2(m)

        out = self.dense_hidden(y)
        out = self.dense_output(out)
        out = self.dropout3(out, training=training)
        out = self.layernorm3(out + y)

        return out
