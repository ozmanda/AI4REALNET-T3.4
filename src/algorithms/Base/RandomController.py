import random
from typing import Dict, Any

class RandomController:
    """
    A controller that randomly selects actions for agents in the environment.
    This is a placeholder implementation for testing purposes.
    """

    def __init__(self, n_actions, n_agents) -> None:
        self.n_actions: int = n_actions
        self.n_agents: int = n_agents

    def act(self) -> Dict[Any, int]:
        """
        Selects a random action for each agent in the environment.
        """
        actions: Dict[Any, int] = {}
        for agent_id in range(self.n_agents):
            actions[agent_id] = random.randint(0, self.n_actions - 1)
        return actions