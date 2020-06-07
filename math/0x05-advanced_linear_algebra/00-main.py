#!/usr/bin/env python3

if __name__ == '__main__':
    determinant = __import__('0-determinant').determinant

    mat0 = [[]]
    mat = [[5, 7, 9], [3, 1], [6, 2, 4]]
    matx = [[0]]
    maty = [[1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
            [1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
            [1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
            [1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
            [1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
            [1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
            [1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
            [1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
            [1, 2, 3, 4, 5, 6, 5, 4, 3, 2],
            [91, 92, 93, 94, 5, 6, 7, 8, 99, 10]]
    matz = None
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]
    mat7 = [[5, 7, 9, 8], [3, 1, 8, 5], [6, 2, 4, 1], [1, 2, 3, 4]]
    mat8 = [[5, 7, 9, 8], (3, 1, 8, 5), [6, 2, 4, 1], [1, 2, 3, 4]]

    print(determinant(mat0))

    try:
        determinant(mat)
    except Exception as e:
        print(e)

    print(determinant(matx))

    try:
        determinant(maty)
    except Exception as e:
        print(e)

    try:
        determinant(matz)
    except Exception as e:
        print(e)

    print(determinant(mat1))
    print(determinant(mat2))
    print(determinant(mat3))
    print(determinant(mat4))
    print(determinant(mat7))

    try:
        determinant(mat8)
    except Exception as e:
        print(e)

    try:
        determinant(mat5)
    except Exception as e:
        print(e)

    try:
        determinant(mat6)
    except Exception as e:
        print(e)
