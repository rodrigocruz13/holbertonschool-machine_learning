#!/usr/bin/env python3


def matrix_transpose(matrix):
    """Matrix transpose function
    """

    column = len(matrix[0])
    row = len(matrix)
    transpose = []
    for j in range(column):
        transpose.append([])
        for i in range(row):
            num = matrix[i][j]
            transpose[j].append(num)
    return transpose
