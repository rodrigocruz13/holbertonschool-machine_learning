#!/usr/bin/env python3
"""
Module used to
"""

import numpy as np


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set:

    Args:
        data is the list of data to calculate the moving average of
        beta is the weight used for the moving average
        Your moving average calculation should use bias correction
    Returns:
        a list containing the moving averages of data
    """

    β = beta
    # F(t) = β * F(t - 1) + (1 - β) * a(t)
    # a = real data
    # F = forcasted data
    # bias_correction = 1 - (β ** (i + 1))

    a_list = []
    data_lenght = len(data)
    moving_avg = 0
    for i in range(data_lenght):
        moving_avg = ((moving_avg * β) + ((1 - β) * data[i]))
        bias_correction = 1 - (β ** (i + 1))
        a_list.append(moving_avg / bias_correction)
    return a_list
