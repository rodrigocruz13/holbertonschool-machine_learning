#!/usr/bin/env python3
"""
Module used to
"""


def dims(matrix):
    """
    Calculates the dimensions of a matrix:

    Arguments
    ---------
    matrix    : list

    Return
    ------
    dims      : list
                list  with the dimensions of the matrix
    """

    dim = [len(matrix)]
    while (isinstance(matrix[0], list)):
        dim.append(len(matrix[0]))
        matrix = matrix[0]
    return dim


def sumation(mat1, mat2):
    """
    Calculates the recursive sum of mat1 and mat2:

    Arguments
    ---------

    - mat1    : list
    - mat2    : list
                mat1 and mat2 have the same shape

    Return
    ------
    sum       : list
                sum of mat1 and mat2
    """
    sum_ = []
    n = len(mat1)
    for i in range(n):
        is_list = True if isinstance(mat1[i], list) else False
        x = sumation(mat1[i], mat2[i]) if (is_list) else mat1[i] + mat2[i]
        sum_.append(x)
    return sum_


def add_matrices(mat1, mat2):
    """
    Calculates the recursive sum of mat1 and mat2:

    Arguments
    ---------

    - mat1    : list
    - mat2    : list
                mat1 and mat2 have the same shape

    Return
    ------
    summation or none if mat1 and mat2 have different shapes
    - sum     : list
                sum of mat1 and mat2
    """

    dim_1 = dims(mat1)
    dim_2 = dims(mat2)

    return (None if (dim_1 != dim_2) else sumation(mat1, mat2))
