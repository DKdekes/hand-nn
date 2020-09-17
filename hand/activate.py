import numpy as np


def relu_single(x):
    return x if x > 0 else 0.001 * x


relu = np.vectorize(relu_single)


def d_relu(x):
    x[x <= 0] = 0.001
    x[x > 0] = 1
    return x
