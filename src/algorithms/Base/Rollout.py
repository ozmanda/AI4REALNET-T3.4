from typing import Tuple
from torch import Tensor


class Rollout(): 
    def __init__(self):
        """
        Initialise the rollout.
        """
        self.transitions = []

    def append_transition(self, transition) -> None:
        """
        Append a transition to the rollout.
        """
        self.transitions.append(transition)

    def is_empty(self) -> bool:
        """
        Check if the rollout is empty.
        """
        return not self.transitions
    
    def unzip_transitions(self) -> Tuple:
        """
        Unzip the transitions into separate tensors.
        """
        pass