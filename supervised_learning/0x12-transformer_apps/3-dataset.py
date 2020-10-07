#!/usr/bin/env python3
""" Module used to """


import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset ():
    """[Loads and preps a dataset for machine translation]

    Returns:
        [type]: [description]
    """

    def __init__(self, batch_size, max_len):
        """[summary]

        Args:
            batch_size ([type]): [batch size for training/validation]
            max_len ([type]): [max num of tokens allowed per example sentence]

        Update the data_train attribute by performing the following actions:
        -   filter out all examples that have either sentence with more than
            max_len tokens.
        -   cache the dataset to increase performance
        -   shuffle the entire dataset
        -   split the dataset into padded batches of size batch_size
        -   prefetch the dataset using tf.data.experimental.AUTOTUNE to
            increase performance

        Update the data_validate attribute by performing the following actions:
        -   filter out all examples that have either sentence with more than
            max_len tokens
        -   split the dataset into padded batches of size batch_size
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

    def encode(self, pt, en):
        """[Instance method encodes a translation into token]

        Args:
            pt ([tf.Tensor]): [Tensor with the Portuguese sentence]
            en ([tf.Tensor]): [Tensor with the corresponding English sentence]

        The tokenized sentences should include the start & end of sentence
        tokens
        The start token should be indexed as vocab_size
        The end token should be indexed as vocab_size + 1

        Returns:
            pt_tokens, en_tokens

            pt_tokens ([tf.Tensor]): [Tensor with the Portuguese tokens]
            en_tokens ([tf.Tensor]): [Tensor with the Portuguese tokens]
        """

        # https://www.tensorflow.org/tutorials/text/transformer

        pt_a = self.tokenizer_pt.vocab_size
        # print(pt_a)

        pt_b = self.tokenizer_pt.encode(pt.numpy())
        # print(pt_b)

        pt_c = pt_a + 1
        # print(pt_c)

        pt_tokens = [pt_a] + pt_b + [pt_c]
        # print(pt_tokens)


        en_a = self.tokenizer_en.vocab_size
        en_b = self.tokenizer_en.encode(pt.numpy())
        en_c = en_a + 1
        en_tokens = [en_a] + en_b + [en_c]

        # print(pt_tokens, en_tokens)

        return pt_tokens, en_tokens
