#!/usr/bin/env python3

# Avoid Tensorflow warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np

SelfAttention = __import__('1-self_attention').SelfAttention

attention = SelfAttention(256)
print(attention.W)
print(attention.U)
print(attention.V)
s_prev = tf.convert_to_tensor(np.random.uniform(
    size=(32, 256)), preferred_dtype='float32')
hidden_states = tf.convert_to_tensor(np.random.uniform(
    size=(32, 10, 256)), preferred_dtype='float32')
context, weights = attention(s_prev, hidden_states)
print(context)
print(weights)
