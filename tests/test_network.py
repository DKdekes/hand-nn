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
    expected = 0.5440674117449611
    res = network.forward_propagate(x)
    assert expected == res[0][0], 'bad forward propogate calculation'


def test_backward_propogate(network):
    x = 1
    label = 0
    network.forward_propagate(x)
    network.backward_propagate(label)
    assert network.model[2].nodes[0].delta == -0.1349603084197159
    assert network.model[1].nodes[0].delta == -0.00484004464901974
    assert network.model[0].nodes[0].delta == -1.0177676543954743e-07


def test_update_weights(network):
    x = 1
    label = 0
    network.forward_propagate(x)
    network.backward_propagate(label)
    assert network.model[2].nodes[0].bias == 0.0923385947687978
    network.update_weights(x)
    assert network.model[2].nodes[0].bias == 0.02485844055893985


def test_train(network):
    pass


