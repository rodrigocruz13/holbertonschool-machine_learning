#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    mean_cov = __import__('0-mean_cov').mean_cov

    try:
        mean, cov = mean_cov(None)
        print(mean)
        print(cov)
    except Exception as e:
        print(e)

    try:
        mean, cov = mean_cov([])
        print(mean)
        print(cov)
    except Exception as e:
        print(e)

    X = np.ndarray(shape=(1, 1), dtype=float, order='F')

    try:
        mean, cov = mean_cov(X)
        print(mean)
        print(cov)
    except Exception as e:
        print(e)

    a = [12, 30, 10]
    b = [[36, -30, 15], [-30, 100, -20], [15, -20, 25]]
    np.random.seed(0)
    X = np.random.multivariate_normal(a, b, 10000)
    mean, cov = mean_cov(X)
    print(mean)
    print(cov)
