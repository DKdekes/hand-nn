import numpy as np


def C(a, y):
    return np.sum(np.square(a - y))


def dC(a, y):
    return np.sum(2 * (a - y))


if __name__ == '__main__':
    pass
