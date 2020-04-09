import pytest
from numpy import random
from hand.layer import DenseLayer
random.seed(1)


@pytest.fixture
def simple_dense_layer():
    return DenseLayer.DenseLayer(1, 1)


@pytest.fixture
def dense_layer():
    return DenseLayer.DenseLayer(3, 3)


def test_compute_simple(simple_dense_layer):
    x = 1
    print([node.w for node in simple_dense_layer.nodes])
    output = simple_dense_layer.compute(x)
    print(output)


def test_compute(dense_layer):
    x = [0.2, 0.4, 0.6]
    expected = [0.6739353958396248, 0.523603469940592, 1.231616397486213]
    res = dense_layer.compute(x)
    assert expected == res
