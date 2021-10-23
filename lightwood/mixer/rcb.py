# Recycling circuit breaker mixer
from lightwood.mixer import BaseMixer
from lightwood.encoder import BaseEncoder
from lightwood.data import EncodedDs
from typing import Dict
from lightwood.helpers.torch import LightwoodAutocast
from torch import nn
import torch
import pandas as pd


class RCBNet(nn.Module):
    no_loops: int
    null_output: torch.Tensor
    device: torch.device
    modules: nn.ModuleList
    input_size: int
    start_grad: int

    def __init__(self, input_size: int, output_size: int, device: torch.device) -> None:
        super(RCBNet, self).__init__()
        self.to(device)
        self.no_loops = 5
        self.null_output = torch.zeros(output_size)
        self.input_size = input_size
        self.start_grad = 0
        modules = []
        for idx in reversed(list(range(1, self.no_loops + 1))):
            layer_in_size = min(input_size, (input_size / self.no_loops) * idx)
            layer_in_size += output_size

            layers = [nn.Linear(layer_in_size, layer_in_size)]
            if idx < self.no_loops:
                layers.append(nn.SELU())

            modules.append(torch.nn.Sequential(layers))

        self.modules = nn.ModuleList(modules)

    def to(self, device: torch.device) -> torch.nn.Module:
        self.modules = self.modules.to(device)
        self.device = device
        return self

    def _grad_loop_bit(self, X):
        start = min(self.input_size + 1, int(self.input_size / self.no_loops) * i)
        end = min(self.input_size + 1, int(self.input_size / self.no_loops) * (1 + i))

        if X is None:
            X = input[start:end] + self.null_output
            X = self.modules[i].forward(X)
        else:
            X = X[:start] + input[start:end] + X[start:]

        # Circuit breaker condition
        if False:
            output = X[end:]
            return output, X
        else:
            return None, X

    def _forward_int(self, input: torch.Tensor):
        for n in self.no_loops:
            X = None
            for i in range(n):
                if self.start_grad >= n:
                    output, X = self._grad_loop_bit(self, X)
                else:
                    with torch.no_grad():
                        output, X = self._grad_loop_bit(self, X)

    def forward(self, input: torch.Tensor):
        try:
            with LightwoodAutocast():
                output = self._forward_int(input)
        except Exception:
            output = self._forward_int(input)

        return output


class RCB(BaseMixer):
    model: nn.Module
    dtype_dict: dict
    target: str
    stable: bool = True

    def __init__(self, stop_after: int, target: str, dtype_dict: Dict[str, str], target_encoder: BaseEncoder, 
                 fit_on_dev: bool):
        super().__init__(stop_after)
        self.dtype_dict = dtype_dict
        self.target = target
        self.target_encoder = target_encoder
        self.fit_on_dev = fit_on_dev

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        pass

    def partial_fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        pass

    def __call__(self, ds: EncodedDs) -> pd.DataFrame:
        pass

