#!/usr/bin/env python3
""" Module used to """


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ class RNNEncoder
        Inherits from tensorflow.keras.layers.Layer to encode 4 ML translation

    Args:
        tf ([type]): [description]
    """

    def __init__(self, vocab, embedding, units, batch):
        """[Class constructor]

        Args:
            vocab       ([int]):    [size of the input vocabulary]
            embedding   ([int]):    [dimensionality of the embedding vector]
            units       ([int]):    [number of hidden units in the RNN cell]
            batch       ([int]):    [batch size instance attributes]
        Attributes:
            batch       Batch size
            units       Number of hidden units in the RNN cell
            embedding   keras Embedding layer that converts words from
                        the vocabulary into an embedding vector
            gru         A keras GRU layer with units units.

        Returns:
            Both the full sequence of outputs as well as the last hidden state
            Recurrent weights should be initialized with glorot_uniform
        """

        # super() function that will make the child class inherit all the
        # methods and properties from its parent:

        init_weights = 'glorot_uniform'
        super(RNNDecoder, self).__init__()

        self.embedding = tf.keras.layers.Embedding(output_dim=embedding,
                                                   input_dim=vocab)

        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer=init_weights,
                                       return_sequences=True,
                                       return_state=True)

        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """[summary]

        Args:
            x       ([tensor]): [is a tensor of shape (batch, input_seq_len)
                                containing the input to the encoder layer as
                                word indices within the vocabulary]
            s_prev  ([tensor]): [tensor of shape (batch, units) containing the
                                 previous decoder hidden state]
            hidden_states ([tensor]): [tensor of shape (batch, input_seq_len,
                                       units) containing the outputs of the
                                       encoder]

        Returns:
            [type]: [description]
        """

        attention = SelfAttention(2048)
        context, weights = attention(s_prev, hidden_states)

        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, 1), x], axis=-1)

        outputs, s = self.gru(inputs=x)
        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)

        return (y, s)
