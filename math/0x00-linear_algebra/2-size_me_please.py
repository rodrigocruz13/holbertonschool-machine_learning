#!/usr/bin/env python3


def matrix_shape(matrix):
    """Matrix shape function
    """

    size = []
    if type(matrix) is list:
        size.append(len(matrix))

        e = matrix[0]
        if type(e) is list:
            matrix_shape_rec(e, size)
    return size


def matrix_shape_rec(matrix, size):
    """Matrix shape function
    """
    if type(matrix) is list:
        size.append(len(matrix))

        e = matrix[0]
        if type(e) is list:
            matrix_shape_rec(e, size)
    return size
