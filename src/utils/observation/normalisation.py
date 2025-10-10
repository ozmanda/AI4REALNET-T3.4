import torch
from torch import Tensor
from typing import Dict, Tuple

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

    def update_metrics(self, x: Tensor) -> None:
        """ Update the running mean and variance using a batch of observations and the RunningMeanStd class. """
        self.rms.update_batch(x)
        self.mean = self.rms.mean.item()
        self.std = torch.sqrt(self.rms.var + self.eps).item()
        self.count = self.rms.count
        
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
        if clip is not None:
            self.clip = clip
        if c is not None:
            self.c = c
            
        x_normalised = (x - self.mean) / (self.std + self.eps)
        if self.clip:
            x_normalised = torch.clamp(x_normalised, -self.c, self.c)
        return x_normalised


class FlatlandNormalisation(Normalisation):
    def __init__(self, n_nodes: int, n_features: int, n_agents: int, env_size: Tuple[int, int], 
                 eps: float = 1e-8, clip: bool = True, c: float = 5.0) -> None:
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
        self.max_distance: int = env_size[0] * env_size[1]  # (width, height)

    def normalise(self, x: Tensor, clip: bool = None, c: float = None) -> Tensor:
        """
        Normalise a batch of observations using the running mean and std for distance features, and normalising the number of agents features using the
        total number of agents in the environment. Assumes an 3D input shape of (batch_size, n_agents, n_nodes * n_features).
        """
        if clip is not None:
            self.clip = clip
        if c is not None:
            self.c = c

        batchsize = x.shape[0]
        x = x.view(batchsize, self.n_agents, self.n_nodes, self.n_features)
        x = self._normalise_distance_metrics(x)
        x = self._normalise_agent_counts(x)
        x = self._normalise_speed(x)
        return x.view(batchsize, self.n_agents, self.n_nodes * self.n_features)

    def _normalise_distance_metrics(self, x: Tensor) -> Tensor:
        """ Normalise the distance features (0:7) using the running mean and std. """
        x[:, :, :, :7] = x[:, :, :, :7] / self.max_distance  # scale distances to [0, 1]
        return x

    # TODO: finish implementing running mean and std for distance metrics over all workers
    # def _normalise_distance_metrics(self, x: Tensor) -> Tensor:
    #     """ Normalise the distance features (0:7) using the running mean and std. """
    #     distance_std = self.distance_rms.std + self.eps
    #     for i in range(self.n_nodes):
    #         x[:, :, i, :7] = (x[:, :, i, :7] - self.distance_rms.mean) / distance_std
    #         if self.clip:
    #             x[:, :, i, :7] = torch.clamp(x[:, :, i, :7], -self.c, self.c)
    #     return x

    def _normalise_agent_counts(self, x: Tensor) -> Tensor:
        """
        Normalise the number of agents features (7-9, 11) by dividing by the total number of agents in the environment.

        Parameters:
            - x: New data points (Tensor of shape (batch_size, n_nodes, n_features))
        
        Returns:
            - x_normalised: Normalised data points (Tensor of shape (batch_size, n_nodes, n_features))
        """
        x[:, :, :, 7:10] = x[:, :, :, 7:10] / self.n_agents   # TODO: check that this works the expected way
        x[:, :, :, 11] = x[:, :, :, 11] / self.n_agents
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


class IdentityNormalisation(Normalisation):
    """
    No-op normalisation that returns observations unchanged. Useful for environments where no specific scaling is required.
    """
    def __init__(self) -> None:
        super().__init__()

    def normalise(self, x: Tensor, clip: bool = None, c: float = None) -> Tensor:
        return x
