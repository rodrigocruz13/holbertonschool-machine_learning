#!/usr/bin/env python3
# Avoid Tensorflow warnings
import tensorflow as tf
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

Dataset = __import__('0-dataset').Dataset

tf.compat.v1.enable_eager_execution()
data = Dataset()
for pt, en in data.data_train.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
for pt, en in data.data_valid.take(1):
    print(pt.numpy().decode('utf-8'))
    print(en.numpy().decode('utf-8'))
print(type(data.tokenizer_pt))
print(type(data.tokenizer_en))
