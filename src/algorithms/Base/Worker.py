from torch import Tensor

class Worker:
    def __init__(self) -> None:
        """ 
        Initialise the worker.
        """
        pass

    def run(self) -> None:
        """
        Run a single episode in the environment and collect rollouts.
        """
        pass

    def _generalised_advantage_estimator(self, state: Tensor, next_state: Tensor, reward: Tensor, done: Tensor, step: int) -> Tensor:
        """
        Calculate the Generalised Advantage Estimator (GAE) for the given state and reward.

        Parameters:
            - state          Tensor      (batch_size, n_features)
            - next_state     Tensor      (batch_size, n_features)
            - reward         Tensor      (batch_size)
            - done           Tensor      (batch_size)
            - step           int         Current step in the episode

        Returns:
            - advantages     Tensor      (batch_size)
        """
        pass