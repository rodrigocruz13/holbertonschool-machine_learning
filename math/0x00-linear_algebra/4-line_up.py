#!/usr/bin/env python3
"""
Module used to add two arrays
"""


def add_arrays(arr1, arr2):
    """Matrix add function
    """
    row = len(arr1)
    new_array = []
    if (len(arr1) != len(arr2)):
        return None

    for element in range(row):
        new_array.append(arr1[element] + arr2[element])
    return new_array
