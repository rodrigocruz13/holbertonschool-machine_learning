#!/usr/bin/env python3
"""
Module used to concatenate 2 arrays
"""


def cat_arrays(arr1, arr2):
    """Matrix concatenate function
    """

    if (type(arr1) is not list) or (type(arr2) is not list):
        return None

    new_array = []
    for i in range(len(arr1)):
        new_array.append(arr1[i])

    for i in range(len(arr2)):
        new_array.append(arr2[i])
    return new_array
