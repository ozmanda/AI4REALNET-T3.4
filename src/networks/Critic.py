import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from typing import List, Dict
from argparse import Namespace

from src.networks.Attention import MultiHeadAttention
from src.networks.Recursive import RecursiveLayer
from src.utils.network_utils import create_linear_layers

class Critic(nn.Module): 
    def __init__(self, n_features: int, config: Dict): 
        super().__init__()
        input_size = n_features * config['n_nodes']
        layers: List[nn.Linear] = create_linear_layers(input_size=input_size, output_size=1, layer_sizes=config['layer_sizes']) # TODO: where are these used?
        self.seq = nn.Sequential(RecursiveLayer(n_features=n_features, output_size=1, n_nodes=config['n_nodes']),)

    def forward(self, state: Tensor):
        return self.seq(state)
        
        