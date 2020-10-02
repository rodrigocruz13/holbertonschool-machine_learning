#!/usr/bin/env python3
""" Module used to """


import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """[calculates the scaled dot product attention:]

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
        if mask is not None, multiply -1e9 to the mask and add it to the
        scaled matrix multiplication

    The preceding dimensions of Q, K, and V are the same

    Returns: output, weights
        outputa [tensor]: [tensor with its last two dimensions as
                           (..., seq_len_q, dv) containing the scaled dot
                           product attention]
        weights [tensor]: [tensor with its last two dimensions as
                          (..., seq_len_q, seq_len_v) containing the
                          attention weights
    """

    # scale matmul_qk with scaled_attention_logits
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    sal = matmul_qk / tf.math.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))

    # add the mask
    sal = sal + (mask * -1e9) if mask is not None else sal

    # softmax
    weights = tf.nn.softmax(sal, axis=-1)
    output = tf.matmul(weights, V)

    return output, weights
