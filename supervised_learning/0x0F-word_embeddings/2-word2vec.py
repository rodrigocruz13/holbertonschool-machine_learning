#!/usr/bin/env python3
"""
1. tf_idf
"""

import gensim as g


def word2vec_model(sentences,
                   size=100,
                   min_count=5,
                   window=5,
                   negative=5,
                   cbow=True,
                   iterations=5,
                   seed=0,
                   workers=1):
    """[Function that creates and trains a gensim word2vec model]

    Args:
        sentences ([list]):         [sentences to be trained on]
        size (int, optional):       [is the dimensionality of the embedding
                                     layer]. Defaults to 100.
        min_count (int, optional):  [minimum number of occurrences of a word
                                     for use in training]. Defaults to 5.
        window (int, optional):     [maximum distance between the current and
                                     predicted word within a sentence].
                                     Defaults to 5.
        negative (int, optional):   [size of negative sampling]. Defaults to 5
        cbow (bool, optional):      [boolean to determine the training type;
                                     True is for CBOW; False is for Skip-gram]
                                     Defaults to True.
        iterations (int, optional): [number of iterations to train over].
                                     Defaults to 5.
        seed (int, optional):       [seed for the random number generator].
                                     Defaults to 0.
        workers (int, optional):    [number of worker threads to train the
                                     model]. Defaults to 1.

    Returns:
        [type]: [the trained model]
    """

    # https://radimrehurek.com/gensim/models/word2vec.html

    sg = 0 if cbow else 1

    model = g.models.Word2Vec(sentences=sentences,
                              size=size,
                              window=window,
                              min_count=min_count,
                              negative=negative,
                              sg=sg,
                              seed=seed,
                              workers=workers,
                              iter=iterations)

    total_examples = model.corpus_count

    # To avoid common mistakes around the model’s ability to do multiple
    # training passes itself, an explicit epochs argument MUST be provided.
    # In the common and recommended case where train() is only called once,
    # you can set epochs=self.iter.
    epochs = model.iter

    model.train(sentences=sentences,
                total_examples=total_examples,
                epochs=epochs)

    return model
