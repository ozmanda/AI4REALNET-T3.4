import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

class FeedForwardNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: Dict) -> None:
        super(FeedForwardNN, self).__init__()
        self.seq = self.make_layers(input_size, output_size, config)

    def make_layers(self, input_size: int, output_size: int, config: Dict) -> nn.Sequential:
        layers = []
        in_size = input_size
        for hidden_size in config['layer_sizes']:
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.ReLU())
            in_size = hidden_size
        layers.append(nn.Linear(in_size, output_size))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)