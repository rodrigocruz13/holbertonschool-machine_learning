#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables needed to calculate the P affinities in t-SNE:

    Args:
    - X         numpy.ndarray       Array of shape (n, d) containing the
                                    dataset to be transformed by t-SNE
                - n     (int)       number of data points
                - d     (int)       number of dimensions in each point
                Note: All dims have a mean of 0 across all data points
    - perplexity                    It is the perplexity that all Gaussian
                                    distributions should have

    Returns:
        - (D, P, betas, H)
        - D     numpy.ndarray       Array of shape (n, n) that calculates
                                    the pairwise distance between two points
        - P     numpy.ndarray       Array of shape (n, n) initialized to all
                                    0‘s that will contain the P affinities
        - betas numpy.ndarray       Array of shape (n, 1) initialized to all
                                    1’s that will contain all of the beta
                                    values, where
                                    β_i = 1 / (2 (σ_i ^2))
        - H                         The Shannon entropy for perplexity
                                    perplexity

        Note:   σ_i is associated with a parameter called perplexity which
                can be loosely interpreted as the number of close neighboor
                each point has
    """

    n = X.shape[0]

    P = np.zeros((n, n))
    betas = np.ones((n, 1))

    # Euclidean distance 'D'
    # https://nlml.github.io/in-raw-numpy/in-raw-numpy-t-sne/
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    H = np.log2(perplexity)

    return D, P, betas, H


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point

    Args:
        - Di:   numpy.ndarray       Arrar of shape (n - 1,) containing the
                                    parwise distances between a data point
                                    and all other points except itself
                - n:      int       Number of data points
        beta:   float               beta value for the Gaussian distribution

    Returns:    (Hi, Pi)
        - Hi:     float             Shannon entropy of the points
        - Pi:     numpy.ndarray     Array of shape (n - 1,) containing the
                                    P affinities of the points
    """

    # Stochastic Neighbor Embedding

    Pi = np.exp(-Di * beta) / np.sum(np.exp(-Di * beta))
    Hi = np.sum(Pi * np.log2(Pi))

    return -Hi, Pi
