import torch
from torch import Tensor
from typing import Tuple, Dict

from src.utils.observation.RunningMeanStd import RunningMeanStd

class Normalisation:
    def __init__(self, eps: float = 1e-8, clip: bool = True, c: float = 5.0) -> None:
        """
        Normalisation for single values, holds a RunningMeanStd instance.

        Parameters:
            - eps: Small value to avoid division by zero
        """
        self.eps: float = eps
        self.mean: float = 0.0
        self.std: float = 0.0
        self.count = 0
        self.rms: RunningMeanStd = RunningMeanStd(size=1, eps=eps)

        self.clip: bool = clip
        self.c: float = c

    def update_metrics(self, x: Tensor) -> Dict[str, Tensor]:
        """ Update the running mean and variance using a batch of observations and the RunningMeanStd class. """
        mean, var, count = self.rms.update_batch(x)
        self.mean = mean.item()
        self.std = torch.sqrt(var).item()
        self.count = count
        
    def normalise(self, x: Tensor, clip: bool = None, c: float = None) -> Tensor:
        """
        Normalise a batch of observations using the running mean and std.

        Parameters:
            - x: New data points (Tensor of shape (batch_size,))
            - clip: Whether to clip the normalised values
            - c: Clip value
        
        Returns:
            - x_normalised: Normalised data points (Tensor of shape (batch_size,))
        """
        if clip:
            self.clip = clip
            self.c = c if c else self.c
            
        x_normalised = (x - self.mean) / (self.std + self.eps)
        if self.clip:
            x_normalised = torch.clamp(x_normalised, -self.c, self.c)
        return x_normalised


class FlatlandNormalisation:
    def __init__(self, n_nodes: int, n_features: int, n_agents: int, eps: float = 1e-8, clip: bool = True, c: float = 5.0) -> None:
        """
        Normalisation for Flatland tree observations, holds a RunningMeanStd instance for each type of observation, as described
        in the Flatland documentation: https://flatland-association.github.io/flatland-book/environment/observation_builder/provided_observations.html
            - features 0:7 are distances, for which a running mean and std is kept
            - features 7:10 and 11 are a number of agents and divided by the total number of agents in the environment
            - feature 10 is a speed and left unchanged

        Parameters:
            - n_nodes: Number of nodes in the tree observation
            - n_features: Number of features per node in the tree observation
            - eps: Small value to avoid division by zero
        """
        super().__init__(eps=eps, clip=clip, c=c)
        self.n_nodes: int = n_nodes
        self.n_features: int = n_features
        self.n_agents: int = n_agents
        self.distance_rms: RunningMeanStd = RunningMeanStd(size=1, eps=eps)  

    def normalise(self, x: Tensor, clip: bool = True, c: float = 5.0) -> Tensor:
        batchsize = x.shape[0]
        x = x.resize((batchsize, self.n_nodes, self.n_features))
        x = self._normalise_distance_metrics(x)
        x = self._normalise_agent_counts(x)
        x = self._normalise_speed(x)
        x = x.resize((batchsize, self.n_nodes * self.n_features))
        return x

    def _normalise_distance_metrics(self, x: Tensor) -> Tensor:
        """ Normalise the distance features (0:7) using the running mean and std. """
        self._update_distance_metrics(x)
        for i in range(self.n_nodes):
            x[:, i, :7] = (x[:, i, :7] - self.distance_rms.mean) / (self.distance_rms.std + self.eps)
            if self.clip:
                x[:, i, :7] = torch.clamp(x[:, i, :7], -self.c, self.c)

    def _update_distance_metrics(self, x: Tensor) -> None:
        """
        Update the running mean and variance for distance features (0-6) using a batch of observations.

        Parameters:
            - x: New data points (Tensor of shape (batch_size, n_nodes, n_features))
        """
        x = x.reshape(-1, self.n_features)
        self.distance_rms.update_batch(x[:, :7])

    def _normalise_agent_counts(self, x: Tensor) -> Tensor:
        """
        Normalise the number of agents features (7-9, 11) by dividing by the total number of agents in the environment.

        Parameters:
            - x: New data points (Tensor of shape (batch_size, n_nodes, n_features))
        
        Returns:
            - x_normalised: Normalised data points (Tensor of shape (batch_size, n_nodes, n_features))
        """
        x[:, :, 7:10] = x[:, :, 7:10] / self.n_agents   # TODO: check that this works the expected way
        x[:, :, 11] = x[:, :, 11] / self.n_agents
        return x
    
    def _normalise_speed(self, x: Tensor) -> Tensor:
        """
        Normalise the speed feature (10) - currently left unchanged.

        Parameters:
            - x: New data points (Tensor of shape (batch_size, n_nodes, n_features))
        
        Returns:
            - x_normalised: Normalised data points (Tensor of shape (batch_size, n_nodes, n_features))
        """
        # TODO: implement a better normalisation for speed
        return x
