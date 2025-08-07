from typing import List, Tuple, Dict

class Runner(): 
    def __init__(self): 
        """
        Initialise the runner.
        """

    def run(self) -> List:
        """
        Run a single episode in the environment and collect rollouts.
        """
        pass

    def _select_actions(self) -> None: 
        """
        Select actions based on the current state of the environment.
        """
        pass

    def _save_transitions(self) -> None:
        """
        Save the transitions from the current episode to the rollouts.
        """
        pass