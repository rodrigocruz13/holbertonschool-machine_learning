#!/usr/bin/env python3
"""
Module used to multiply 2 2D arrays
"""


def mat_mul(mat1, mat2):
    """
    Matrix add function
    """

    if (not isinstance(mat1, list) or not isinstance(mat2, list)):
        return None

    if (len(mat1[0]) != len(mat2)):
        return None

    new = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            row.append(0)
        new.append(row)

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                new[i][j] += mat1[i][k] * mat2[k][j]
    return new
