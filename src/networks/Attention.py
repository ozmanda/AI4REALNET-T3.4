'''
Adapted from the JBR_HSE Solution to the Flatland 2020 Challenge at NeurIPS
Original Code: https://github.com/jbr-ai-labs/NeurIPS2020-Flatland-Competition-Solution
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MultiHeadAttention(nn.Module):
    """
    MultiHeadAttention class as described in the original paper "Attention is All You Need" by Vaswani et al: 
    https://arxiv.org/abs/1706.03762
    """
    def __init__(self, n_features: int, num_heads: int) -> None:
        super().__init__()
        self.n_heads: int = num_heads
        #? what does this scale metric do?
        self.scale = 1.0 / n_features ** 0.5

        #? define what these layers are exactly and how they interact
        self.fc_key = nn.Linear(n_features, n_features * self.n_heads, bias=False)
        self.fc_query = nn.Linear(n_features, n_features * self.n_heads, bias=False)
        self.fc_value = nn.Linear(n_features, n_features * self.n_heads, bias=False)

    #? do we want to keep or rename query and key
    def forward(self, key: Tensor, query: Tensor, value: Tensor) -> Tensor:
        batchsize: int = key.size(0)
        len_key: int = key.size(1)
        len_query: int = query.size(1)
        len_value: int = value.size(1)
        n_features: int = key.size(2)

        key: Tensor = self.fc_key(key).view(batchsize, len_key, self.n_heads, n_features)
        query: Tensor = self.fc_query(query).view(batchsize, len_query, self.n_heads, n_features)
        value: Tensor = self.fc_value(value).view(batchsize, len_value, self.n_heads, n_features)

        #? look at all of this in more detail
        key, query, value = key.transpose(1, 2), query.transpose(1, 2), value.transpose(1, 2)
        weights = F.softmax(torch.matmul(query, key.transpose(2, 3)) * self.scale, dim=-1)
        output = torch.matmul(weights, value)
        output = torch.flatten(output.transpose(1, 2), start_dim=2)
        return output
