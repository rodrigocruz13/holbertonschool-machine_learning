#!/usr/bin/env python3
""" Module used to """


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
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
        super(RNNEncoder, self).__init__()

        self.batch, self.units = batch, units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       recurrent_initializer=init_weights,
                                       return_sequences=True,
                                       return_state=True)

    def initialize_hidden_state(self):
        """[Public instance method]
            Initializes the hidden states for the RNN cell to a tensor of 0's

            Returns: a tensor of shape (batch, units) containing the
            initialized hidden states
        """

        return tf.keras.initializers.Zeros()(shape=(self.batch, self.units))

    def call(self, x, initial):
        """[Public instance method]

        Args:
            x       ([tensor]): [is a tensor of shape (batch, input_seq_len)
                                containing the input to the encoder layer as
                                word indices within the vocabulary]
            initial ([tensor]): [tensor of shape (batch, units) containing the
                                initial hidden state]
        Returns:
            outputs, hidden

            outputs ([tensor]): Tensor of shape (batch, input_seq_len, units)
                                containing the outputs of the encoder
            hidden  ([tensor]): Tensor of shape (batch, units) containing the
                                last hidden state of the encoder
        """

        outputs, hidden = self.gru(self.embedding(x), initial_state=initial)

        return outputs, hidden
