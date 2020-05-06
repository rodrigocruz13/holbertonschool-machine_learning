#!/usr/bin/env python3
""" module """


def aux_det_mat(matrix, mul):
    """
    Auxiliar function that calculate the determinand of a matrix by
    by accumulatin values
    Args:
        - matrix:       list of lists whose determinant should be calculated
        - mul:          constant value to multiplicate the matrix
    Returns:
            The value of the determinant
    """

    width = len(matrix)
    if width == 1:
        return mul * matrix[0][0]
    else:
        sign = -1
        deter = 0
        for i in range(width):
            m = []
            for j in range(1, width):
                buff = []
                for k in range(width):
                    if k != i:
                        buff.append(matrix[j][k])
                m.append(buff)
            sign *= -1
            deter += mul * aux_det_mat(m, sign * matrix[0][i])
        return deter


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
    if len(matrix) != len(matrix[0]) and len(matrix[0]):
        raise ValueError('matrix must be a square matrix')

    return aux_det_mat(matrix, 1)
