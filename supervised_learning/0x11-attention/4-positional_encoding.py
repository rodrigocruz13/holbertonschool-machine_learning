#!/usr/bin/env python3
""" Module used to """


import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """[calculates the positional encoding for a transformer]

    Args:
        max_seq_len     ([int]):    [maximum sequence length]
        dm              ([type]):   [model depth]
    Returns:
        pev:            ([array]):   array of shape (max_seq_len, dm)
                                    containing the positional encoding vectors
    """

    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = 2 * (np.arange(dm)[np.newaxis, :]//2) / np.float32(dm)

    pev = pos / np.power(10000, i)

    # Applying SIN to odd indices
    pev[:, 0::2] = np.sin(pev[:, 0::2])

    # Applying COS to odd indices
    pev[:, 1::2] = np.cos(pev[:, 1::2])

    return pev
