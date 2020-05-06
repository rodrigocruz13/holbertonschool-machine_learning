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


def recursive_determinant(matrix, total=0):
    """ calculates recursively the determinant of a matrix
    Args:
        matrix - is a list of lists whose determinant should be calculated
        total - summatory of determinants
    Returns:
        the determinant of 2D matrix
    """
    # Extract all indices of matrix
    indices = list(range(len(matrix)))

    # This method works recursively, always we gonna calculate the
    # determinant of a 2D matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    for index in indices:
        cp_matrix = matrix.copy()
        # Remove the first column
        cp_matrix = cp_matrix[1:]

        rows_length = len(cp_matrix)

        for i in range(rows_length):
            # Removes column
            cp_matrix[i] = cp_matrix[i][0:index] + cp_matrix[i][index + 1:]

        # Change the sign of all pairs indices
        sign = (-1) ** (index % 2)

        sub_det = recursive_determinant(cp_matrix)

        total += sign * matrix[0][index] * sub_det

    return total


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
    if len(matrix) != len(matrix[0]):
        raise ValueError('matrix must be a square matrix')

    return recursive_determinant(matrix)
