import numpy as np


def s(x, c=1):
    return 1 / (1 + np.exp(-c * x))


def ds(x, c=1):
    return s(x, c) * (1 - s(x, c))

# good test
def test_s():
    for i in range(-4, 4):
        print(i)
        print(s(i, 3))
        print()
        print()


if __name__ == '__main__':
    test_s()