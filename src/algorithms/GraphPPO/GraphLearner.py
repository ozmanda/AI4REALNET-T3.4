import torch
from typing import Dict

class GraphLearner():
    """
    This class performs learning on a graph-representation of Flatland environments.
    """
    def __init__(self, config: Dict):
        self.config = config