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

    # there is no matrix
    if not matrix:
        raise TypeError('matrix must be a list of lists')

    # the shell is not a list
    if not isinstance(matrix, list):
        raise TypeError('matrix must be a list of lists')

    # each row has to be a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError('matrix must be a list of lists')

    # matrix with one row but that row is empty
    if len(matrix) == 1 and len(matrix[0]) == 0:
        return 1

    # list with one row and 1 element
    if len(matrix) == 1 and len(matrix[0]) == 1:
        return matrix[0][0]

    # matrix wih size m x n
    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError('matrix must be a square matrix')

    return aux_det(matrix, len(matrix))


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
    Calculates the minor matrix of a matrix:
    Args:
    - matrix:       list of lists whose minor matrix should be calculated
            1. If matrix is not a list of lists, raise a TypeError with the
               message matrix must be a list of lists
            2. If matrix is not square or is empty, raise a ValueError with
               the message matrix must be a non-empty square matrix
    Returns:
        the minor matrix of matrix
    """

    # 1. There is no matrix
    if not matrix:
        raise TypeError('matrix must be a list of lists')

    # 1. The shell is not a list
    if not isinstance(matrix, list):
        raise TypeError('matrix must be a list of lists')

    # 1. Each row has to be a list
    for row in matrix:
        if not isinstance(row, list):
            raise TypeError('matrix must be a list of lists')

    # 2.validate is not square
    for row in matrix:
        if len(matrix) != len(row):
            raise ValueError('matrix must be a non-empty square matrix')

    # 2. Validate is not empty
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
