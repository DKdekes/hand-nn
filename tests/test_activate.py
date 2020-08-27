import pytest
import numpy as np

np.random.seed(1)
from hand.activate import relu, d_relu


def test_relu():
    x = np.random.rand(3, 3) - 0.2
    print(relu(x))


def test_d_relu():
    x = np.random.rand(3, 3) - 0.2
    print(d_relu(x))