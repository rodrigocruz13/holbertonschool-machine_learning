#!/usr/bin/env python3
"""
1. tf_idf
"""

import tensorflow.keras as K


def gensim_to_keras(model):
    """[Function that converts a gensim word2vec model 2 keras Embedding layer]

    Args:
        model ([type]): [description]

    Returns:
        [type]: [description]
    """

    # https://radimrehurek.com/gensim/models/word2vec.html

    k_embedding_layer = model.wv.get_keras_embedding(train_embeddings=True)

    return k_embedding_layer
