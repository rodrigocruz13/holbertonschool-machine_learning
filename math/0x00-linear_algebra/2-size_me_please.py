#!/usr/bin/env python3
"""
Module used to find the size of a matrix
"""


def matrix_shape(matrix):
    """
    Matrix shape function
    """

    size = []
    if isinstance(matrix, list):
        size.append(len(matrix))

        e = matrix[0]
        if isinstance(e, list):
            matrix_shape_rec(e, size)
    return size


def matrix_shape_rec(matrix, size):
    """
    Matrix shape recursive function
    """
    if isinstance(matrix, list):
        size.append(len(matrix))

        e = matrix[0]
        if isinstance(e, list):
            matrix_shape_rec(e, size)
    return size
