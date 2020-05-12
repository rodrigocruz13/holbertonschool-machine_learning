#!/usr/bin/env python3

if __name__ == '__main__':
    import numpy as np
    from multinormal import MultiNormal

    a = [12, 30, 10]
    b = [[36, -30, 15], [-30, 100, -20], [15, -20, 25]]
    np.random.seed(0)
    data = np.random.multivariate_normal(a, b, 10000).T
    mn = MultiNormal(data)
    print(mn.mean)
    print(mn.cov)
