import pytest
import numpy as np
np.random.seed(1)
from hand.network.Network import Network

 
@pytest.fixture
def network_simple():
    return Network(1, [1], 1)


@pytest.fixture
def network():
    return Network(3, [3], 1)


def test_forward_propagate_simple(network_simple):
    x = 1
    expected = 0.5440674117449611
    res = network_simple.forward_propagate(x)
    assert expected == res[0][0], 'bad forward propogate calculation'


def test_forward_propogate(network):
    x = [1, 0, 1]
    '''
    layer 0
        weights:
            [0.417022]
        bias:
            0.7203244934421581
        weights:
            [0.00011437]
        bias:
            0.30233257263183977
        weights:
            [0.14675589]
        bias:
            0.0923385947687978
    layer 1
        weights:
            [0.18626021 0.34556073 0.39676747]
        bias:
            0.538816734003357
        weights:
            [0.41919451 0.6852195  0.20445225]
        bias:
            0.8781174363909454
        weights:
            [0.02738759 0.67046751 0.4173048 ]
        bias:
            0.5586898284457517
    layer 2
        weights:
            [0.14038694 0.19810149 0.80074457]
        bias:
            0.9682615757193975
    '''
    network.report_weights()

def test_backward_propogate_simple(network_simple):
    x = 1
    label = 0
    network_simple.forward_propagate(x)
    network_simple.backward_propagate(label)
    assert network_simple.model[2].nodes[0].delta == -0.1349603084197159
    assert network_simple.model[1].nodes[0].delta == -0.00484004464901974
    assert network_simple.model[0].nodes[0].delta == -1.0177676543954743e-07


def test_update_weights_simple(network_simple):
    x = 1
    label = 0
    network_simple.forward_propagate(x)
    network_simple.backward_propagate(label)
    assert network_simple.model[2].nodes[0].bias == 0.0923385947687978
    network_simple.update_weights(x)
    assert network_simple.model[2].nodes[0].bias == 0.02485844055893985


def test_train(network_simple):
    pass


