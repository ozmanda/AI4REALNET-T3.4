import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List
from argparse import Namespace

from src.networks.Attention import MultiHeadAttention
from src.networks.Recursive import RecursiveLayer
from src.utils.network_utils import create_linear_layers

class Critic(nn.Module): 
    def __init__(self, n_features: int, n_actions: int, layer_sizes: List[int]): 
        super().__init__()
        layers: List[nn.Linear] = create_linear_layers(input_size=n_features, output_size=1, layer_sizes=layer_sizes) #? where are these used?
        self.seq = nn.Sequential(RecursiveLayer(n_features, 1))

    def forward(self, state: Tensor):
        return self.seq(state)
        
        