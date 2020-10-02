#!/usr/bin/env python3

# Avoid Tensorflow warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention

mha = MultiHeadAttention(512, 8)
print(mha.dm)
print(mha.h)
print(mha.depth)
print(mha.Wq)
print(mha.Wk)
print(mha.Wv)
print(mha.linear)
f = 'float32'
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype(f))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype(f))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype(f))
output, weights = mha(Q, K, V, None)
print(output)
print(weights)
