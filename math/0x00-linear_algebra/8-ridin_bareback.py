#!/usr/bin/env python3


def mat_mul(mat1, mat2):
    """Matrix add function
    """

    if (type(mat1) is not list or type(mat2) is not list):
        return None

    if (len(mat1[0]) != len(mat2)):
        return None

    new = []
    for i in range(len(mat1[0])):
        row = []
        for j in range(len(mat1[0])):
            row.append(0)
        new.append(row)

    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                print(mat1[i][k], " x ", mat2[k][j])
                new[i][j] += mat1[i][k] * mat2[k][j]
            print(new[i][j])
    return new