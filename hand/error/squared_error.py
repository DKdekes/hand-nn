import numpy as np


def error(y, y_p):
    if y.shape == ():
        y = np.array([y])
    if y_p.shape == ():
        y_p = np.array([y_p])
    return np.sum(np.square(y - y_p))


if __name__ == '__main__':
    pass
