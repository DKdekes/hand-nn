import torch
import math


class Initializer:
    def __init__(self):
        self.var_scaler = None

    def initialize(self, n_inputs, n_units):
        var_scaler, initialized_tensor = self._initialize(n_inputs, n_units)
        self.var_scaler = var_scaler
        return initialized_tensor

    def _initialize(self, n_inputs, n_units) -> (float, torch.Tensor):
        """
        implement. return (var_scaler, initialized tensor)
        """
        raise Exception('not implemented')


class He(Initializer):

    def _initialize(self, n_inputs, n_units) -> (float, torch.Tensor):
        # init = torch.nn.init.kaiming_normal_(torch.zeros(n_inputs, n_units))
        # init = (2 / n_units) * torch.randn(n_inputs, n_units)
        var_scaler = 2 / (n_inputs)
        init = torch.randn(n_inputs, n_units) * math.sqrt(var_scaler)
        self.var_scaler = var_scaler
        assert math.isclose(init.mean(), 0, abs_tol=0.2)
        assert math.isclose(init.var(), var_scaler, abs_tol=0.2)
        return var_scaler, init


class Zeros(Initializer):
    def _initialize(self, n_inputs, n_units) -> (float, torch.Tensor):
        return 0, torch.zeros(n_inputs, n_units)


def __getattr__(item):
    for a in globals():
        if a.lower() == item.lower():
            return globals()[a]