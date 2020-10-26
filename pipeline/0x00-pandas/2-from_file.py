#!/usr/bin/env python3

"""
Script that preprocess data for the forecasting the value of BTC:
"""

import numpy as np
import pandas as pd


def from_file(filename, delimiter):
    """[summary]

    Args:
        filename ([type]): [description]
        delimiter ([type]): [description]
    """

    # print(filename, delimiter)

    pf = pd.read_csv(filepath_or_buffer=filename, sep=delimiter)

    return pf
