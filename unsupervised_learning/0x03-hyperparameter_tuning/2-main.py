#!/usr/bin/env python3

import numpy as np
GP = __import__('2-gp').GaussianProcess


def f(x):
    """our 'black box' function"""
    return np.sin(5 * x) + 2 * np.sin(-2 * x)


if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2 * np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_new = np.random.uniform(-np.pi, 2 * np.pi, 1)
    print('X_new:', X_new)
    Y_new = f(X_new)
    print('Y_new:', Y_new)

    print('\n*******************')
    gp.update(X_new, Y_new)

    print('\nX_new:')
    print(gp.X.shape)
    print(gp.X)

    print('\nY_new:')
    print(gp.Y.shape)
    print(gp.Y)

    print('\nk_new:')
    print(gp.K.shape)
    print(gp.K)
