import numpy as np


def error(o, t):
    return np.sum(0.5 * np.square(o - t))


if __name__ == '__main__':
    pass
