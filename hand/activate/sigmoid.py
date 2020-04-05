import numpy as np


def s(x, c=1):
    return 1 / (1 + np.exp(-c * x))


def ds(x, c=1):
    return x * (1 - c*x)


if __name__ == '__main__':
    pass
