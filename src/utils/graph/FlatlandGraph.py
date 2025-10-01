from typing import Dict, Tuple, Union, List

import networkx as nx
from src.utils.graph.MultiDiGraphBuilder import MultiDiGraphBuilder
from src.utils.graph.PathGenerator import PathGenerator

from flatland.envs.rail_env import RailEnv
from flatland.envs.agent_utils import EnvAgent

class FlatlandGraph:
    def __init__(self, env: RailEnv):
        self.env: RailEnv = env
        self.graph: nx.MultiDiGraph
        self.station_lookup: Dict[Tuple[int, int], Union[str, int]]
        self.station_nodes: Dict[str, Tuple[int, int]]

        # initialisation functions
        self._init_graph()

    def _init_agent_dict(self) -> None:
        pass

    def _init_graph(self) -> None:
        builder = MultiDiGraphBuilder(self.env)
        self.graph = builder.graph
        self.station_lookup = builder.station_lookup
        self.station_nodes = builder.stations

    def _init_paths(self, agents: List[EnvAgent], k: int = 4, weight: str = 'length', weight_goal: str = 'min'):
        """
        Initialize the path generator to compute k-shortest paths between stations.

        Parameters:
            - agents: List[EnvAgent], list of agents in the environment
            - k: int, number of shortest paths to compute between each station pair
            - weight: str, edge attribute to use as weight for path calculation
            - weight_goal: str, whether to minimize or maximize the weight ('min' or 'max')

        Returns:
            - TBD
        """
        pass
