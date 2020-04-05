import numpy as np


def sigmoid(x, c=1):
    return 1 / (1 + np.exp(-c * x))


def d_sigmoid(x, c=1):
    return sigmoid(x) * (1 - sigmoid(x, c))


if __name__ == '__main__':
    pass
