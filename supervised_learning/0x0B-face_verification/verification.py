#!/usr/bin/env python3
"""
class FaceAlign
"""

import numpy as np
import tensorflow.keras as K


class FaceVerification():
    """ Class constructor """

    def __init__(self, model, database, identities):
        """
        Class initializator
        Args:
            - model:        face verification embedding model or the path to
                            where the model is stored
            - database:     numpy.ndarray of all the face embeddings in the
                            database
            - identities:   list corresponding to the embeddings in the DB
        """

        with K.utils.CustomObjectScope({'tf': tf}):
            self.model = K.models.load_model(model)

        self.database = database
        self.identities = identities

    def embedding(self, images):
        """
        Function that get the embeddings:
        Args:
            -  images:      images to retrieve the embeddings of

        Returns: a numpy.ndarray of embeddingse
        """

        return self.model.predict(images)
