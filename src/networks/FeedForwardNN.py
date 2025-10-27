import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

class FeedForwardNN(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: Dict) -> None:
        super(FeedForwardNN, self).__init__()
        self.hidden_layers = nn.ModuleList()
        in_size = input_size
        for hidden_size in config['layer_sizes']:
            self.hidden_layers.append(nn.Linear(in_size, hidden_size))
            in_size = hidden_size
        self.output_layer = nn.Linear(in_size, output_size)
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize network weights using orthogonal initialization for better gradient flow."""
        for layer in self.hidden_layers:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.constant_(layer.bias, 0.0)
        
        nn.init.orthogonal_(self.output_layer.weight, gain=0.01)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)