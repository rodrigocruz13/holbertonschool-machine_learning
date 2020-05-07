#!/usr/bin/env python3
""" module """


def aux_det(matrix, n):
    """
    Auxiliar function that calculate the determinand of a matrix by
    by accumulatin values
    Args:
        - matrix:       list of lists whose determinant should be calculated
        - m:            constant value to multiplicate the matrix
    Returns:
            The value of the determinant
    """

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    n = len(matrix)
    ret = 0
    for i in range(n):
        if i % 2 == 0:
            ret += matrix[0][i] * aux_det([row[:i] + row[i + 1:]
                                           for row in matrix[1:]], n - 1)
        else:
            ret -= matrix[0][i] * aux_det([row[:i] + row[i + 1:]
                                           for row in matrix[1:]], n - 1)
    return ret


def determinant(matrix):
    """
    Calculates the determinant of a matrix:
    Args:
        - matrix:       list of lists whose determinant should be calculated
                - If matrix is not a list of lists, raise a TypeError with
                  the message matrix must be a list of lists.
                - If matrix is not square, raise a ValueError with the message
                  matrix must be a square matrix
                The list [[]] represents a 0x0 matrix
    Returns:
        - the determinant of matrix
    """

    # validate list of list
    _is = isinstance
    if not _is(matrix, list):
        raise TypeError('matrix must be a list of lists')

    if not matrix:
        raise TypeError('matrix must be a list of lists')

    for row in matrix:
        if not _is(row, list):
            raise TypeError('matrix must be a list of lists')

    # list with one row but that row is empty
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # list with one row and 1 element
    if len(matrix[0]) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    # matrix wih size m x n
    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError('matrix must be a square matrix')

    if len(matrix) == 1:
        return matrix[0][0]

    return aux_det(matrix, len(matrix))
