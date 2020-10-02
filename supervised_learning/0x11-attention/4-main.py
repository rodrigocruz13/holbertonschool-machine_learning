#!/usr/bin/env python3

# Avoid Tensorflow warnings
from warnings import simplefilter
simplefilter(action='ignore', category=FutureWarning)

import numpy as np
positional_encoding = __import__('4-positional_encoding').positional_encoding

PE = positional_encoding(30, 512)
print(PE.shape)
print(PE)
