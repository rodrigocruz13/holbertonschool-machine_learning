#!/usr/bin/env python3

# Avoid Tensorflow warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention

f = 'float32'
np.random.seed(0)
Q = tf.convert_to_tensor(np.random.uniform(size=(50, 10, 256)).astype(f))
K = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 256)).astype(f))
V = tf.convert_to_tensor(np.random.uniform(size=(50, 15, 512)).astype(f))
output, weights = sdp_attention(Q, K, V)
print(output)
print(weights)
