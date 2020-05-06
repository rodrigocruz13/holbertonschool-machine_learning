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

    if not _is(matrix, list) or not any(_is(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    if len(matrix[0]) == 0:
        return 1

    if len(matrix[0]) == 1:
        return matrix[0][0]

    if len(matrix) != len(matrix[0]) and len(matrix[0]):
        raise ValueError('matrix must be a square matrix')

    return aux_det_mat(matrix, 1)


def create_mini_matrix(matrix, i, j):
    """
        Generates a copy of all the elements of matrix of size [n][n] except
        those that are located in row i, column j

        Args:
            - matrix:   matrix
            - i:        row to be excluded from the copy
            - j:        column to be exclude from the copy

        Returns:
            A copy of size [n-1][n-1] of the matrix
    """
    copy = []

    r = 0
    for row in matrix:
        copy_row = []
        if r != i:
            c = 0
            for col in row:
                if (c != j):
                    copy_row.append(col)
                c = c + 1
            copy.append(copy_row)
        r = r + 1
    return copy


def minor(matrix):
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
    if not _is(matrix, list) or not any(_is(row, list) for row in matrix):
        raise TypeError('matrix must be a list of lists')

    # validate is not square or is empty
    if len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a non-empty square matrix')

    # validate is not empty
    if (matrix == [] or matrix[0] == []):
        raise ValueError('matrix must be a non-empty square matrix')

    # calculate minor for each cell of the matrix
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return [[1]]

    minor = []
    for i in range(len(matrix)):
        minor_row = []
        for j in range(len(matrix)):
            copy_matrix = create_mini_matrix(matrix, i, j)
            det = determinant(copy_matrix)
            minor_row.append(det)
        minor.append(minor_row)
    return minor
