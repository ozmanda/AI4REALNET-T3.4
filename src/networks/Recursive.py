import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from src.utils.observation.obs_utils import get_depth


class RecursiveLayer(nn.Module): 
    def __init__(self, n_features: int, output_size: int, n_nodes: int, hidden_size=64): # TODO: refactor to use n_nodes
        super().__init__()
        self.n_features = n_features 
        self.input_size = n_features * n_nodes # number of features per node
        self.n_nodes = n_nodes
        self.hidden_size = hidden_size
        self.feature_encoding = nn.Sequential(nn.Linear(self.input_size, hidden_size), 
                                              nn.ReLU(inplace=True),
                                              nn.Linear(hidden_size, 2 * hidden_size),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(2 * hidden_size, 3 * hidden_size),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(3 * hidden_size, hidden_size),
                                              nn.ReLU(inplace=True),
                                              nn.Linear(self.hidden_size, output_size) 
                                              )
        
    def forward(self, state: Tensor) -> Tensor:
        """
        Forward pass of the Recursive layer. Has been changed from the original JBR_HSE version to be strongly simplified, as the recursive structure was not working.

        Parameters:
            - state         Tensor      (batch_size, n_features*n_nodes)
        """
        return self.feature_encoding(state) # (batch_size, output_size)
    

    def forward_old(self, state: Tensor): 
        """
        Forward pass. Assumes that the input state is a tensor of shape (batch_size, n_features + general_size), where the "general features" begin at position 0 and end at position general_size - 1. The remaining features are the "state features".

        Parameters: 
            - state         Tensor      (batch_size, n_features * n_nodes)

        Returns
            - output        Tensor      (batch_size, output_size)
        """
        batch_size = state.size(0)
        features = state.view(batch_size*self.n_nodes, self.n_features) # (batch_size, n_nodes, n_features)
        encoded_features = self.feature_encoding(features) # (batch_size, n_nodes, hidden_size)
        encoded_features = encoded_features.view(batch_size, self.n_nodes, self.hidden_size)
        depth = get_depth(state.size(1))

        # merge parent and child features and pass through merged features encoding
        merged = [torch.zeros(batch_size, 4 ** (level + 1), self.hidden_size) for level in range(depth + 1)] # contains children features of level i at index i

        # TODO: expand kTree to make this general and include state features
        # iterates through the tree levels from the bottom to the top, merging the features of the current level with its children
        for level in range(depth, 0, -1):
            parent_start: int = ((4 ** level - 1) // (4 - 1)) - 1
            parent_end: int = ((4 ** (level + 1) - 1) // (4 - 1)) - 1

            current_features = encoded_features[:, parent_start:parent_end, :]
            children_features = merged[level].view(batch_size, 4 ** level, -1)

            merged_features = torch.cat([current_features, children_features], dim=2).view(-1, 2 * self.hidden_size)
            merged[level - 1] = self.merged_feature_encoding(merged_features) 

        values = merged[0].view(batch_size, 2 * self.hidden_size)

        # pass through final network layer
        output: Tensor = self.final(values)
        return output