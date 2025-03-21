import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

def create_linear_layers(input_size: int, output_size: int, layer_sizes: List[int]) -> List[nn.Linear]: 
    """
    Creates a list of sequential linear layers with ReLU activation function #? output nn.Sequential?

    Parameters: 
        - input_size        int         size of the first input
        - output_size       int         desired size of the output of the last layer
        - layer_size        List[int]   list of layer sizes 

    Output: 
        - layers            List        list of linear layers with activation functions
    """
    layers: List[nn.Linear] = []
    
    for layer_size in layer_sizes:
        layers.append(nn.Linear(input_size, layer_size))
        layers.append(nn.ReLU(inplace=True))
        input_size = layer_size
    
    layers.append(nn.Linear(input_size, output_size))
    return layers