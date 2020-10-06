#!/usr/bin/env python3
""" Module used to """


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset ():
    """[Loads and preps a dataset for machine translation]

    Returns:
        [type]: [description]
    """

    def __init__(self):
        """[Class constructor. Creates the instance attributes]

        Attributes:
            data_train      contains the ted_hrlr_translate/pt_to_en
                            tf.data.Dataset train split, loaded as_supervided
            data_valid      contains the ted_hrlr_translate/pt_to_en
                            tf.data.Dataset validate split, loaded
                            as_supervided
            tokenizer_pt    is the Portuguese tokenizer created from the
                            training set
            tokenizer_en    is the English tokenizer created from the
                            training set
        """

        # https://www.tensorflow.org/tutorials/text/transformer

        ds = 'ted_hrlr_translate/pt_to_en'
        tr = 'train'
        vl = 'validation'

        t_sample, v_sample = tfds.load(ds, with_info=True, as_supervised=True)
        self.data_train, self.data_valid = t_sample[tr], t_sample[vl]

        portuguese, english = self.tokenize_dataset(self.data_train)
        self.tokenizer_pt, self.tokenizer_en = portuguese, english

    def tokenize_dataset(self, data):
        """[Instance method that creates sub-word tokenizers for our dataset]

        Args:
            data ([tf.data.Dataset]):   [Dataset whose examples are formatted
                                        as a tuple (pt, en)]
                pt tf.Tensor with the Portuguese sentence
                en tf.Tensor with the corresponding English sentence
        Notes:
            The maximum vocab size should be set to 2**15
        Returns:
            tokenizer_pt, tokenizer_en [type]:
            [tokenizer_pt is the Portuguese tokenizer]
            [tokenizer_en is the English tokenizer]
        """

        # https://www.tensorflow.org/tutorials/text/transformer

        size = (2 ** 15)
        corpus = tfds.features.text.SubwordTextEncoder.build_from_corpus

        pt = corpus((pt.numpy() for pt, en in data), target_vocab_size=size)
        en = corpus((en.numpy() for pt, en in data), target_vocab_size=size)

        return pt, en
