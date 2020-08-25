import numpy as np


def squared_error(y, y_p):
    if isinstance(y, int) or y.shape == ():
        y = np.array([y])
    if isinstance(y_p, int) or y_p.shape == ():
        y_p = np.array([y_p])
    return np.sum(np.square(y - y_p))


def d_squared_error(y, y_p):
    if isinstance(y, int) or y.shape == ():
        y = np.array([y])
    if isinstance(y_p, int) or y_p.shape == ():
        y_p = np.array([y_p])
    return np.sum(2 * (y - y_p))


if __name__ == '__main__':
    y_ = 1
    y_p_ = 2
    squared_error(y_, y_p_)