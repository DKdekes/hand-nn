import pytest
import numpy as np
np.random.seed(1)
from hand.network import network


def test_forward_propagate():
    print()
    net = network.Network(1, [1], 1)
    print(net.report_weights())



