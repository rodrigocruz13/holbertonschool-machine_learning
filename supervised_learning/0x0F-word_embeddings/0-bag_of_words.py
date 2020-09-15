#!/usr/bin/env python3
"""
0. Bag Of Words
"""

import numpy as np


def bag_of_words(sentences, vocab=None):
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

    features = []
    alt_sentences = []
    if not (isinstance(sentences, list)):
        return None, None

    for i in range(len(sentences)):
        sentence = sentences[0].lower().split()
        alt_words = []

        for word in sentence:
            word = word.split("'")[0]
            w = ""
            for letter in word:
                if (letter >= 'a' and letter <= 'z'):
                    w += letter
            if (w not in features):
                features.append(w)
            alt_words.append(w)

        alt_sentences.append(alt_words)
        sentences.append(sentences[0].lower().split("!")[0].split("'")[0])
        sentences.pop(0)
    features.sort()
    features = np.array(features)

    embeddings = []
    for sentence in alt_sentences:
        embedding = [0] * len(features)
        for word in sentence:
            for j, feature in enumerate(features):
                if feature in sentence:
                    embedding[j] = 1
        embeddings.append(embedding)
    embeddings = np.array(embeddings)

    return embeddings, features
