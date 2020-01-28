#!/usr/bin/env python3
"""
Module used to concatenate 2 2D arrays
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """Matrix concatenate function
    """

    new = []
    copy_m1 = [row[:] for row in mat1]
    copy_m2 = [row[:] for row in mat2]

    if (len(mat1[0]) == len(mat2[0])) and (axis == 0):
        new = copy_m1 + copy_m2
    elif (len(mat1) == len(mat2)) and (axis == 1):
        for i in range(len(mat1)):
            new.append(mat1[i] + mat2[i])
    else:
        return None
    return new
