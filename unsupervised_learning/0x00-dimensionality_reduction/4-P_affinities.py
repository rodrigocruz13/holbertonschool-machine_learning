#!/usr/bin/env python3
"""
PCA: principal components analysis
"""

import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Write a function def P_affinities(X, tol=1e-5, perplexity=30.0): that
    calculates the symmetric P affinities of a data set:

    Arguments:
    X:          numpy.ndarray of shape (n, d) containing the dataset
                to be transformed by t-SNE
      - n       is the number of data points
      - d       is the number of dimensions in each point
    perplexity  is the perplexity that all Gaussian distributions
                should have
    tol        is the maximum tolerance allowed (inclusive) for the diff in
                Shannon entropy from perplexity for all Gaussian distributions
    Returns:
    P, a numpy.ndarray of shape (n, n) containing the symmetric
    P affinities
    """

    # symbols:  ≈ Π Σ ² ³ σ ∩ ∪ α β γ δ ε ζ η θ Φ º ~ Δ π σ ± ⋅ ÷ ∫ ∝ ∞ ⊆ ⊂ ≠

    # source:
    # https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding
    # section Software and then to: https://lvdmaaten.github.io/tsne/ -> python

    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D, P, β, logU = P_init(X, perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Compute the Gaussian kernel and entropy for the current precision
        βmin = -np.inf
        βmax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))]
        (H, thisP) = HP(Di, β[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        # tries = 0
        # while np.abs(Hdiff) > tol and tries < 50:
        while np.abs(Hdiff) > tol:

            # If not, increase or decrease precision
            inf_range = (np.inf, -np.inf)
            if Hdiff > 0:
                βmin = β[i].copy()
                β[i] = β[i] * 2 if (βmax in inf_range) else (β[i] + βmax) / 2
            else:
                βmax = β[i].copy()
                β[i] = β[i] / 2 if (βmin in inf_range) else (β[i] + βmin) / 2.

            # Recompute the values
            (Hi, thisP) = HP(Di, β[i])
            Hdiff = Hi - logU
            # tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i + 1:n]))] = thisP

    # Convrting P into a symmetric matrix
    P = (P + P.T) / (2 * n)
    return P
