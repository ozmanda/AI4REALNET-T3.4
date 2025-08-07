from src.algorithms.Base.Worker import Worker
from src.algorithms.Base.Rollout import Rollout

class Learner():
    def __init__(self) -> None:
        """
        Initialise the learner.
        """
        pass

    def _optimise(self, rollouts) -> None:
        """
        Optimise the model parameters using the collected rollouts.
        """
        pass

    def _loss(self) -> None:
        """
        Calculate the loss for the model.
        """
        pass

    def rollouts(self) -> None:
        """
        Run the PPO algorithm for a specified number of iterations, collecting rollouts and optimising the model.
        """
        pass