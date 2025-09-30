import os
from typing import Tuple, Dict, Any, Union, List
from flatland.envs.rail_env import RailEnv
from src.utils.graph.MultiDiGraphBuilder import MultiDiGraphBuilder



class ConflictPredictor(): 
    """
    Class that predicts conflicts in a railway environment using its MultiDiGraph representation.
    """
    def __init__(self, env: RailEnv):
        """
        Initialize the ConflictPredictor with a given environment.

        Parameters:
            - env: RailEnv, the railway environment to analyze
        """
        self.env = env
        self.graph = MultiDiGraphBuilder(env)

    def _init_conflict_predictor(self):
        self._place_agents()


    def _place_agents(self):
        for agent_handle in self.env.get_agent_handles():
            agent = self.env.agents[agent_handle]
            if self.env.agents[agent_handle].position is not None:
                position: Tuple[int, int] = self.env.agents[agent_handle].position
            else: 
                position: Tuple[int, int] = self.env.agents[agent_handle].initial_position

            # get initial origin node
            self._get_origin_node(position, agent_handle)

            if self.graph.rail_clusters[position[0], position[1]] > 0: 
                self.env.agents[agent_handle].rail_ID = self.graph.rail_clusters[position[0], position[1]]
            elif self.graph.switch_clusters[position[0], position[1]] > 0:
                self.env.agents[agent_handle].rail_ID = self.graph.switch_clusters[position[0], position[1]]
            else: 
                raise ValueError(f"Agent {agent_handle} is not placed on a valid rail or switch.")
            
            
    def _match_edges(self) -> None:
        """ Match edges in the graph to the agents' rail IDs. """
        for agent_handle in self.env.get_agent_handles():
            rail_ID = self.env.agents[agent_handle].rail_ID
            if rail_ID is not None:
                for u, v, data in self.graph.graph.edges(data=True):
                    if data.get('rail_ID') == rail_ID:
                        self.env.agents[agent_handle].rail_edge = (u, v)
                        break


    def _get_origin_node(self, position: Tuple[int, int], agent_handle: Union[int, str]) -> None: 
        """ Add the agent's origin node. """
        pass


    def _get_station_paths(self) -> None:
        """
        Calculate all possible paths through a graph between two stations
        """
        pass


    def cell_time_window(self, cell: Tuple[int, int], train_speed: float) -> Tuple[int, int]:
        """
        Calculate the time window for a given cell based on the train's speed.

        Parameters:
            - cell: Tuple[int, int], the cell coordinates (x, y)
            - train_speed: float, the speed of the train

        Returns:
            - Tuple[int, int]: the time window (start_time, end_time)   
        """
        pass

    
    def edge_time_window(self, edge: Tuple[Any, Any], train_speed: float) -> Tuple[int, int]:
        """
        Calculate the time window for a given edge based on the train's speed.

        Parameters:
            - edge: Tuple[Any, Any], the edge represented by its two nodes (u, v)
            - train_speed: float, the speed of the train

        Returns:
            - Tuple[int, int]: the time window (start_time, end_time)
        """
        pass


    def path_time_windows(self, path: list, train_speed: float) -> List[Tuple[int, int]]:
        """
        Calculate the time windows for a given path based on the train's speed.

        Parameters:
            - path: list, the sequence of nodes representing the path
            - train_speed: float, the speed of the train

        Returns:
            - List[Tuple[int, int]]: the list of time windows (start_time, end_time) for each segment of the path
        """
        pass

    
    def _calculate_conflict_time(self, path_A: List[Tuple[int, int]], path_B: List[Tuple[int, int]],
                                 speed_A: float, speed_B: float, t_A: float, t_B: float) -> bool:
        """
        Determine whether two agents with planned paths A and B, speeds s_A, s_B and departure times 
        t_A and t_B will conflict on overlapping paths.
        """
        pass


    def conflict_matrix(self):
        """
        Calculate the conflict matrix for the paths of all station pairs
        """
        pass