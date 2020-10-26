#!/usr/bin/env python3

"""
Script that preprocess data for the forecasting the value of BTC:
"""

import numpy as np
import pandas as pd


def from_numpy(array):
    """[summary]

    Args:
        array ([np.ndarray]): [array from which you should create the
                               pd.DataFrame]
        The columns of the pd.DataFrame should be labeled in alphabetical
        order and capitalized. There will not be more than 26 columns.
    Returns: the newly created pd.DataFrame

    """

    if (array is None):
        return None

    lenght = len(array[0])

    # Uppercase alphabet
    letters = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
    letters = letters[0: lenght]

    df = pd.DataFrame(data=array, columns=letters)

    return df
