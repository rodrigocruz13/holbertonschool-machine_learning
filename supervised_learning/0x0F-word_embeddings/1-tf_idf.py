#!/usr/bin/env python3
"""
1. tf_idf
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """[Function that creates a bag of words embedding matrix:]

    Args:
        sentences ([list]): [list of sentences to analyze]
        vocab ([list], optional): [list of the vocabulary words to use for the
                                   analysis]. Defaults to None.

    Returns: embeddings, features
        embeddings [np.ndarray]: [array of shape (s, f) with the embeddings]
            s is the number of sentences in sentences
            f is the number of features analyzed
        features [np.ndarray]: [list of the features used for embeddings]
    """

    # sklearn.feature_extraction.text.TfidfVectorizer
    # https://bit.ly/3kkbNqn

    vectorizer = TfidfVectorizer(vocabulary=vocab)
    embeddings = vectorizer.fit_transform(sentences).toarray()
    features = vectorizer.get_feature_names()

    return embeddings, features
