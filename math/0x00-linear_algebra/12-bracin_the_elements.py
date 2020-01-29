#!/usr/bin/env python3
"""
Module used to use elementwise operations
"""


def np_elementwise(mat1, mat2):
    """Matrix elementwise operations
    """
    m_sum = mat1 + mat2
    m_sub = mat1 - mat2
    m_mul = mat1 * mat2
    m_div = mat1 / mat2

    return (m_sum, m_sub, m_mul, m_div)
