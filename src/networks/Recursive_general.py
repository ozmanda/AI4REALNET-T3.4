import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from src.utils.observation.obs_utils import get_depth


class RecursiveLayer(nn.Module): 
    def __init__(self, n_features, output_size, hidden_size=64, general_size=0):
        super().__init__()
        self.general_size = general_size
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.feature_encoding = nn.Sequential(nn.Linear(n_features, hidden_size), nn.ReLU(inplace=True))
        self.encode_merged_features = nn.Sequential(nn.Linear(3 * hidden_size, 3 * hidden_size),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(3 * hidden_size, hidden_size),
                                                    nn.ReLU(inplace=True)
                                                    )
        
        self.encode_general = nn.Sequential(nn.Linear(self.general_size, 2 * hidden_size),
                                            nn.ReLU(inplace=True),
                                            nn.Linear(2 * hidden_size, hidden_size),
                                            nn.ReLU(inplace=True)
                                            )

        self.final = nn.Sequential(nn.Linear(2 * hidden_size + hidden_size, 128), 
                                   nn.ReLU(inplace=True),
                                   nn.Linear(128, output_size) 
                                   )

    def forward(self, state: Tensor): 
        """
        Forward pass. Assumes that the input state is a tensor of shape (batch_size, n_features + general_size), where the "general features" begin at position 0 and end at position general_size - 1. The remaining features are the "state features".

        Parameters: 
            - state         Tensor      (batch_size, n_features + general_size)
            - general_size  int         Number of general features
        """
        batch_size = state.size(0)
        
        # encode state features
        features: Tensor = state[:, self.general_size:]
        features = features.view(batch_size, -1, self.n_features)
        encoded_features: Tensor = self.feature_encoding(features)
        
        # Encode general features
        general: Tensor =  state[:, :self.general_size]
        general = self.encode_general(general)
        
        # merge parent and child features and pass through merged features encoding
        depth = get_depth(features.size(1))
        merged = [torch.zeros(batch_size, 2 ** (level + 1), self.hidden_size) for level in range(depth + 1)]

        # TODO: update this to a quaternary tree
        for level in range(depth, 0, -1):
            left_parent_index = 2 ** level - 2
            right_parent_index = 2 ** (level + 1) - 2
            left_child_index = 2 ** (level + 1) - 2
            right_child_index = 2 ** (level + 2) - 2

            current_features = features[:, left_parent_index:right_parent_index]
            children_features = features[:, left_child_index:right_child_index]

            merged[level - 1] = self.encode_merged_features(torch.cat([current_features, children_features], dim=2))

        values = merged[0].view(batch_size, 2 * self.hidden_size)

        # pass through final network layer
        output: Tensor = self.final(torch.cat([general, values]))
        return output