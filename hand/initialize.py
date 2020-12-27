import torch
import math


def he(n_inputs, n_units):
    # init = torch.nn.init.kaiming_normal_(torch.zeros(n_inputs, n_units))
    # init = (2 / n_units) * torch.randn(n_inputs, n_units)
    init = torch.randn(n_inputs, n_units) / math.sqrt(n_inputs)

    # assert math.isclose(init.mean(), 0, abs_tol=0.2), init.mean()
    # assert math.isclose(init.var(), 2 / n_units, abs_tol=0.2), init.var()
    return init


def zeros(inputs, units):
    return torch.zeros(inputs, units)
