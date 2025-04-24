import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from src.utils.obs_utils import get_depth


class RecursiveLayer(nn.Module): 
    def __init__(self, n_features, n_nodes, output_size, hidden_size=64): # TODO: refactor to use n_nodes
        super().__init__()
        self.n_features = n_features # number of features per node
        self.n_nodes = n_nodes # number of nodes in the tree
        self.input_size = n_features * n_nodes # total number of features
        self.hidden_size = hidden_size
        self.feature_encoding = nn.Sequential(nn.Linear(self.input_size, hidden_size), nn.ReLU(inplace=True))
        self.merged_feature_encoding = nn.Sequential(nn.Linear(2 * hidden_size, 3 * hidden_size),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(3 * hidden_size, hidden_size),
                                                    nn.ReLU(inplace=True)
                                                    )

        self.final = nn.Sequential(nn.Linear(3 * hidden_size, 128), 
                                   nn.ReLU(inplace=True),
                                   nn.Linear(128, output_size) 
                                   )

    def forward(self, state: Tensor): 
        """
        Forward pass. Assumes that the input state is a tensor of shape (batch_size, n_features + general_size), where the "general features" begin at position 0 and end at position general_size - 1. The remaining features are the "state features".

        Parameters: 
            - state         Tensor      (batch_size, n_features)

        Returns
            - output        Tensor      (batch_size, output_size)
        """
        batch_size = state.size(0)
        features = state.view(batch_size, -1, self.n_features) # (batch_size, n_nodes, n_features)
        encoded_features = self.feature_encoding(features) # (batch_size, n_nodes, hidden_size)

        # merge parent and child features and pass through merged features encoding
        depth = get_depth(state.size(1))
        merged = [torch.zeros(batch_size, 4 ** (level + 1), self.hidden_size) for level in range(depth + 1)]

        # TODO: expand kTree to make this general and include state features
        # iterates through the tree levels from the bottom to the top, merging the features of the current level with its children
        for level in range(depth, 0, -1):
            parent_start: int = ((4 ** depth - 1) // (4 - 1)) - 1
            parent_end: int = ((4 ** (depth + 1) - 1) // (4 - 1)) - 1
            child_start: int = ((4 ** (depth + 1) - 1) // (4 - 1)) - 1
            child_end: int = ((4 ** (depth + 2) - 1) // (4 - 1)) - 1

            current_features = encoded_features[:, parent_start:parent_end, :]
            children_features = encoded_features[:, child_start:child_end, :]

            merged[level - 1] = self.merged_feature_encoding(torch.cat([current_features, children_features], dim=2)) # TODO: check that this doesn't have variable size (pretty sure it does!)

        values = merged[0].view(batch_size, 2 * self.hidden_size)

        # pass through final network layer
        output: Tensor = self.final(values)
        return output