import numpy as np


def relu(x: np.array):
    return x * (x > 0)


def d_relu(x):
    x[x <= 0] = 0
    x[x > 0] = 1
    return x
