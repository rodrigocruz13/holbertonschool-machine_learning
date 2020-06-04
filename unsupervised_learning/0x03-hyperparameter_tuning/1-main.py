#!/usr/bin/env python3

import numpy as np
GP = __import__('1-gp').GaussianProcess


def f(x):
    """our 'black box' function"""
    return np.sin(5 * x) + 2 * np.sin(-2 * x)


if __name__ == '__main__':
    np.random.seed(0)
    X_init = np.random.uniform(-np.pi, 2 * np.pi, (2, 1))
    Y_init = f(X_init)

    gp = GP(X_init, Y_init, l=0.6, sigma_f=2)
    X_s = np.random.uniform(-np.pi, 2 * np.pi, (10, 1))
    mu, sig = gp.predict(X_s)
    print(mu.shape)
    print(mu)
    print(sig.shape)
    print(sig)
