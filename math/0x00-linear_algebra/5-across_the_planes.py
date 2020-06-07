#!/usr/bin/env python3
"""
Module used to add 2D matrices
"""


def add_matrices2D(mat1, mat2):
    """Matrix add function
    """

    if (not isinstance(mat1, list)) or (not isinstance(mat2, list)):
        return None

    row1 = len(mat1)
    col1 = len(mat1[0])
    row2 = len(mat2)
    col2 = len(mat2[0])

    if (row1 != row2) or (col1 != col2):
        return None

    new_matrix = []
    for r_i in range(row1):
        row_i = []
        for c_i in range(col1):
            row_i.append(mat1[r_i][c_i] + mat2[r_i][c_i])
        new_matrix .append(row_i)

    return new_matrix
