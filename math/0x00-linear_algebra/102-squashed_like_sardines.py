#!/usr/bin/env python3
"""
Module used to
"""


def shape(matrix):
    """
    Calc the shape of a matrix

    Arguments
    ---------
    matrix1      : list

    Returns
    -------
    s            : list
                   listh with the shape of matrix
    """

    s = []
    try:
        n = len(matrix)
        s.append(n)
        if (n):
            return s + shape(matrix[0])

    except BaseException:
        pass
    return s


def recursive(mat1, mat2, axis):
    """
    Concatenate mat1 and mat2 recursiely along axis

    Arguments
    ---------
    mat1      : list
                first matrix
    mat2      : list
                second matrix
    axis      : int
                axis where is going to be concatenated mat1 and mat2

    Returns
    -------

    m3 or None
    m3        : list
                list with mat1 and mat2 concatenated
    """
    new = []
    if (axis == 0):
        new += [i1 for i1 in mat1] + [i2 for i2 in mat2]

    else:
        for i in range(len(mat1)):
            new.append(recursive(mat1[i], mat2[i], axis - 1))

    return new


def cat_matrices(mat1, mat2, axis=0):
    """
    Concatenates two matrices along a specific axis:

    Arguments
    ---------
    mat1      : list
                first matrix
    mat2      : list
                second matrix
    axis      : int
                axis where is going to be concatenated mat1 and mat2

    Returns
    -------

    new or None
    new       : list
                list with mat1 and mat2 concatenated
    """

    shape1, shape2 = (shape(mat1), shape(mat2))

    if (len(shape1) < axis) or (len(shape2) < axis):
        return None

    shape1[axis] = shape2[axis] = 0

    return recursive(mat1, mat2, axis) if (shape1 == shape2) else None
