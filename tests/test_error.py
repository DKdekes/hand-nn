import pytest
from hand import error
from numpy import random
import torch
import math
random.seed(1)


def test_mse_forward():
    mse = error.Mse()
    inp = torch.Tensor([[0.5], [0.6], [0.7]])
    target = torch.Tensor([[0], [1], [0]])
    res = mse(inp, target)
    assert math.isclose(res, 0.3, abs_tol=0.01), res
    return mse


def test_mse_backward():
    mse = test_mse_forward()
    mse.backward()
