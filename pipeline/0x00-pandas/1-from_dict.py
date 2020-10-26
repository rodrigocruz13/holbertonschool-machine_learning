#!/usr/bin/env python3

"""
Script that preprocess data for the forecasting the value of BTC:
"""

import numpy as np
import pandas as pd


dict = {
    "First": [0.0, 0.5, 1.0, 1.5],
    "Second": ['one', 'two', 'three', 'four']
}

index = ["A", "B", "C", "D"]

df = pd.DataFrame(data=dict, index=index)
