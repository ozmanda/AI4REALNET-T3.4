import torch
from torch import Tensor
from typing import Tuple


class RunningMeanStd:
    def __init__(self, size: int=1, eps: float = 1e-8) -> None:
        """
        Update running mean and variance using Welford's algorithm for a single feature. 

        Parameters:
            - size: Number of features (for shape of mean/var tensors)
            - eps: Small value to avoid division by zero
        """
        self.count: int = 0
        self.mean: Tensor = torch.zeros(size)
        self.var: Tensor = torch.zeros(size)
        self.std: Tensor = torch.zeros(size)
        self.M2: Tensor = torch.zeros(size)
        self.eps: float = eps

    def update_batch(self, x: Tensor) -> None:
        """
        Update running mean and variance for a batch, outputting a single mean and var.

        Parameters:
            - x: New data points (Tensor of shape (batch_size, feature_size))

        Returns:
            - mean: Updated mean (Tensor of shape (1,))
            - var: Updated variance (Tensor of shape (1,))
            - count: Updated count of data points (int)
        """
        x = x.reshape(-1)
        batch_count = x.size(0)
        new_count = self.count + batch_count
        batch_mean = torch.mean(x)
        M2_batch = ((x - batch_mean) ** 2).sum()
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / new_count
        self.M2 += M2_batch + delta ** 2 * self.count * batch_count / new_count
        self.count = new_count
        self.var = self.M2 / (self.count + self.eps)
        self.std = torch.sqrt(self.var + self.eps)

    def update_one(self, x: Tensor) -> None:
        """
        Update running mean and variance for a single observation, outputting a single mean and var.

        Parameters:
            - x: New data point (Tensor of any shape, will be averaged)

        Returns:
            - mean: Updated mean (Tensor of shape (1,))
            - var: Updated variance (Tensor of shape (1,))
            - count: Updated count of data points (int)
        """
        x_scalar = torch.mean(x)
        self.count += 1
        delta = x_scalar - self.mean
        self.mean += delta / self.count
        delta2 = x_scalar - self.mean
        self.M2 += delta * delta2
        self.var = self.M2 / (self.count + self.eps)
        self.std = torch.sqrt(self.var + self.eps)

    def update(self, x: Tensor) -> None:
        """ Wrapper for update_one and update_batch, depending on input shape. """
        if x.dim() == 1:
            self.update_one(x)
        elif x.dim() == 2:
            self.update_batch(x)
        else:
            raise ValueError("Input tensor must be 1D or 2D.")
        
    def update_with_metrics(self, batch_mean: float, batch_M2: float, batch_count: int) -> None:
        """
        Update running mean and variance using pre-computed batch statistics.

        Parameters:
            - mean: Mean of the new batch (float)
            - var: Variance of the new batch (float)
            - count: Count of the new batch (int)
        """
        new_count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / new_count
        self.M2 += batch_M2 + delta ** 2 * self.count * batch_count / new_count
        self.count = new_count
        self.var = self.M2 / (self.count + self.eps)
        self.std = torch.sqrt(self.var + self.eps)


class FeatureRunningMeanStd(RunningMeanStd): 
    def __init__(self, n_features: int, eps: float = 1e-8) -> None:
        """
        Normalisation for Flatland tensors where each feature is treated independently.

        Parameters:
            - x: New data points (Tensor of shape (batch_size, feature_size)
            - mean: Current mean (Tensor of shape (feature_size,)
            - var: Current variance (Tensor of shape (feature_size,)
            - count: Current count of data points (int)
            - max_depth: Maximum depth of the tree observation
            - n_nodes: Number of nodes in the tree observation
        """
        super().__init__(n_features, eps)


    def update_batch(self, x: Tensor) -> None:
        """
        Update running mean and variance for each feature in a tensor.

        Parameters:
            - x: New data points (Tensor of shape (batch_size, feature_size)
            
        Returns:
            - mean: Updated mean (Tensor of shape (feature_size,)
            - var: Updated variance (Tensor of shape (feature_size,)
            - count: Updated count of data points (int)
        """
        batch_count = x.size(0)
        new_count = self.count + batch_count
        batch_mean = torch.mean(x, dim=0)
        M2_batch = ((x - batch_mean) ** 2).sum(dim=0)
        delta = batch_mean - self.mean
        self.mean += delta * batch_count / new_count
        self.M2 += M2_batch + delta ** 2 * self.count * batch_count / new_count
        self.count = new_count
        self.var = self.M2 / (self.count + self.eps)
        self.std = torch.sqrt(self.var + self.eps)


    def update_one(self, x: Tensor) -> None:
        """
        Update running mean and variance for each feature in a tensor.

        Parameters:
            - x: New data point (Tensor of shape (feature_size,)
            
        Returns:
            - mean: Updated mean (Tensor of shape (feature_size,)
            - var: Updated variance (Tensor of shape (feature_size,)
            - count: Updated count of data points (int)
        """
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.var = self.M2 / (self.count + self.eps)
        self.std = torch.sqrt(self.var + self.eps)

    def update(self, x: Tensor) -> None:
        """ Wrapper for update_one and update_batch, depending on input shape. """
        if x.dim() == 1:
            self.update_one(x)
        elif x.dim() == 2:
            self.update_batch(x)
        else:
            raise ValueError("Input tensor must be 1D or 2D.")

class FlatlandNormMetrics:
    def __init__(self):
        self.observation = RunningMeanStd()