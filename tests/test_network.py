import pytest
import numpy as np
np.random.seed(1)
from hand.network.Network import Network


@pytest.fixture
def network():
    return Network(1, [1], 1)


def test_forward_propagate(network):
    x = 1
    '''
    layer 0
        weights:
            0.417022004702574
        bias:
            0.7203244934421581
    layer 1
        weights:
            [0.00011437]
        bias:
            0.30233257263183977
    layer 2
        weights:
            [0.14675589]
        bias:
            0.0923385947687978
    '''
    network.report_weights()
    # expected =



