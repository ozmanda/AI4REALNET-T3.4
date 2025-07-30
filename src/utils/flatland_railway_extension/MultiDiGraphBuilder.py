import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, DefaultDict, Any
from collections import defaultdict

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_position_equal, fast_argmax, fast_count_nonzero

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from src.utils.flatland_railway_extension.RailroadSwitchCluster import RailroadSwitchCluster
from src.environments.env_small import small_flatland_env
from flatland.envs.rail_env_action import RailEnvActions

n_directions: int = 4  # range [1:infty)
rail_actions: Dict[int, RailEnvActions] = {
    0: "North",
    1: "East",
    2: "South",
    3: "West",
}

default_edge_attributes: DefaultDict[str, Any] = defaultdict(lambda: 'No Attribute Set')

class MultiDiGraphBuilder:
    def __init__(self, env: RailEnv):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.env: RailEnv = env
        self._init_parameters()
        self._init_switch_clustering()
        self._generate_graph()

    def _init_parameters(self):
        self.height: int = self.env.height
        self.width: int = self.env.width
        self.max_depth: int = self.height * self.width * 10
        self.from_directions: List[int] = [i for i in range(n_directions)]

        self.nodes: Dict[str, Tuple[int, int]] = {}
        self.edges: Dict[str, Tuple[Tuple[int, int], Tuple[int, int]]] = {}

    def _init_switch_clustering(self):
        """Initialize the switch and rail clustering. """
        switch_analyser: RailroadSwitchAnalyser = RailroadSwitchAnalyser(
            self.env,
            handle_diamond_crossing_as_a_switch=True,
            handle_dead_end_as_a_switch=True
        )
        self.switch_clusters: RailroadSwitchCluster = RailroadSwitchCluster(switch_analyser)
        self.rail_clusters: np.ndarray = self.switch_clusters.connecting_edge_cluster_grid
        self.switch_clusters: np.ndarray = self.switch_clusters.railroad_switch_cluster_grid

    def _generate_graph(self) -> None:
        start_node, start_direction = None, None
        for x, y in ((x, y) for x in range(self.height) for y in range(self.width)):
            _, nonzero_directions = self._get_valid_transitions((x, y)) 
            for direction in nonzero_directions:
                start_node, _, _, _, dead_end = self._find_next_node((x, y), direction)
                if start_node and not dead_end:
                    break
            if start_node:
                break

        self._traverse_grid(start_node)
    
    
    def _traverse_grid(self, start_node: Tuple[int, int]):
        """Traverse the grid from the start node in the specified direction."""
        self._add_node(start_node)
        self._current_depth: int = 0
        _, nonzero_transitions = self._get_valid_transitions(start_node)
        for direction in nonzero_transitions:
            self._node_splitter(start_node, direction)


    def _node_splitter(self, node: Tuple[int, int], travel_direction: int) -> None:
        """ Splits the nodes, following the valid transitions. """
        # rotate the out_direction of the previous cell by 180° to be the incoming direction in the current cell
        in_direction: int = self._reverse_direction(travel_direction)
        self._current_depth += 1
        if self._current_depth > self.max_depth:
            raise ValueError("Maximum depth exceeded while traversing the graph.")
        
        # Get valid transitions from the current node
        _, nonzero_directions = self._get_valid_transitions(node)
        for travel_direction in nonzero_directions:
            # if travel_direction == in_direction: # prevents going back to the previous node
                # continue
            next_node, travel_directions, rail_ID, resources, dead_end = self._find_next_node(node, travel_direction)
            edge_attr: Dict[str, Any] = {
                'rail_ID': rail_ID,
                'out_direction': travel_directions[0],
                'in_direction': travel_directions[1],
                'resources': resources, 
                'length': len(resources),
                'max_speed': None
            }
            if next_node:
                self._add_edge(node, next_node, attr=edge_attr)
                if f'{next_node[0]}_{next_node[1]}' not in self.nodes.keys():
                    self._add_node(next_node, dead_end=dead_end)
                    if not dead_end:
                        self._node_splitter(next_node, travel_direction=travel_directions[1])
                    else:
                        # add opposite edge
                        edge_attr['out_direction'] = self._reverse_direction(edge_attr['out_direction'])
                        edge_attr['in_direction'] = self._reverse_direction(edge_attr['in_direction'])
                        edge_attr['resources'] = resources[::-1] 
                        self._add_edge(next_node, node, attr=edge_attr)
                        


    def _find_next_node(self, previous_position: Tuple[int, int], travel_direction: int) -> Tuple[int, Tuple[int, int], int, List[Tuple[Tuple[int, int], int]], bool]:
        """
        Find the next node in the graph based on the current position and direction of traversal.

        Parameters:
            - previous_position: Tuple[int, int], the position of the previous node
            - travel_direction: int, the direction of travel, so the direction at which the train left the previous cell (0: North, 1: East, 2: South, 3: West)

        Returns:
            - next_position: Tuple[int, int], the position of the next node
            - out_direction: int, the direction of travel at the next node
            - rail_ID: int, the ID of the rail cluster at the next node, is None if the next node is a switch
            - dead_end: bool, whether the next node is a dead end
        """
        # first transition to new cell
        resources: List[Tuple[Tuple[int, int], int]] = []
        out_direction: int = travel_direction
        n_transitions = 2
        depth = 0
        current_position = get_new_position(previous_position, travel_direction)
        rail_ID = self.rail_clusters[current_position[0], current_position[1]]
        
        while n_transitions == 2:
            valid_transitions, _ = self._get_valid_transitions(current_position)
            transitions: Tuple = self.env.rail.get_transitions(*current_position, travel_direction)
            n_transitions: int = np.sum(valid_transitions)

            if n_transitions == 2: 
                # continue in the new travel direction
                resources.append((current_position, travel_direction))
                travel_direction = fast_argmax(transitions)
                current_position: Tuple = get_new_position(current_position, travel_direction)
                depth += 1
            else: 
                if depth == 0:
                    rail_ID = None
                if n_transitions == 0:
                    raise ValueError("No valid transitions found at the current position.")
                elif n_transitions > 2: # multiple transitions possible = switch                
                    return current_position, (out_direction, travel_direction), rail_ID, resources, False
                elif n_transitions == 1:  # only one transition possible = dead end
                    return current_position, (out_direction, travel_direction), rail_ID, resources, True

            if depth > self.max_depth:
                raise ValueError("Maximum depth exceeded while finding next node.")                
            

    def _get_valid_transitions(self, position: Tuple[int, int]) -> Tuple[List[int], List[int]]:
        """ 
        Get valid transition directions from the current position.
          - valid_transitions: List indicating if a transition is valid (1) or not (0), for each direction / orientation, [North, East, South, West].
          - nonzero_transitions: List of indices where valid transitions are found, where: 0 = North, 1 = East, 2 = South, 3 = West.
        """
        valid_transitions = [0] * 4 
        for direction in self.from_directions:
            transitions: List = list(self.env.rail.get_transitions(*position, direction))
            valid_transitions = [a or b for a, b in zip(valid_transitions, transitions)]
        nonzero_transitions = [i for i, val in enumerate(valid_transitions) if val != 0]
        return valid_transitions, nonzero_transitions


    def _add_node(self, node_position: Tuple[int, int], dead_end: bool = False):
        """Add a node to the graph."""
        node_id: str = f"{node_position[0]}_{node_position[1]}"
        self.nodes[node_id] = node_position
        self.graph.add_node(node_position, position=node_position, dead_end=dead_end)


    def _add_edge(self, u: Tuple[int, int], v: Tuple[int, int], attr=None):
        """Add an edge to the graph."""
        # TODO: add track number to the edge attributes
        suffix = f"_{attr['rail_ID']}" if attr and attr['rail_ID'] is not None else ''
        edge_ID: str = f"{u}_{v}{suffix}"
        if not edge_ID in self.edges:
            self.graph.add_edge(u, v, attr=attr) 
            self.edges[edge_ID] = (u, v)

    def get_graph(self):
        """Return the constructed graph."""
        return self.graph

    def clear(self):
        """Clear the graph."""
        self.graph.clear()

    def render(self, savepath: str = None):
        """Render the graph showing nodes and edges."""
        dead_end_status = nx.get_node_attributes(self.graph, 'dead_end')
        node_colours = ['lightgray' if dead_end else 'lightblue' for dead_end in dead_end_status.values()]

        connection_style = f'arc3,rad=0.15'

        plt.figure()
        pos = nx.shell_layout(self.graph)
        nx.draw_networkx_nodes(self.graph, pos, node_size=400, node_color=node_colours)
        nx.draw_networkx_labels(self.graph, pos, font_size=8)
        for u, v, k in self.graph.edges(keys=True):
            nx.draw_networkx_edges(self.graph, pos, edgelist=[(u, v)], width=1, edge_color='black', connectionstyle=f'arc3,rad={(k-1)*0.2}')


        nx.draw_networkx_edges(self.graph, pos, edge_color='black', connectionstyle=connection_style)

        # edge_labels = {(u, v, k): f'{k}' for u, v, k in self.graph.edges(keys=True)}
        edge_labels = {
            (u, v, k): f"{d['attr']['rail_ID'] if d['attr']['rail_ID'] is not None else ''}"  # Default to '' if 'rail_ID' is None
            for u, v, k, d in self.graph.edges(keys=True, data=True)
        }
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_color='red', font_size=8)
        # nx.draw(self.graph, 
        #         pos=positions,
        #         labels={node: node for node in self.graph.nodes()},
        #         node_color=node_colours,

        #         )
        plt.axis('off')
        plt.show()
        if savepath:
            plt.savefig(savepath, bbox_inches='tight')
            
        plt.close()
        # nx.draw(
        #     self.graph, pos=positions, edge_color='black', width=1, linewidths=1,
        #     node_size=200, node_color='lightgray', alpha=1.0, font_size=8,
        #     labels={node: node for node in self.graph.nodes()}
        # )

        # nx.draw_networkx_edge_labels(
        #     self.graph, pos=positions,
        #     edge_labels={(u, v): f"{d['key']}" for u, v, d in self.graph.edges(data=True)},
        #     font_color='red', font_size=8
        # )

    def _reverse_direction(self, direction: int) -> int:
        """Reverse the direction."""
        return (direction + (n_directions // 2)) % n_directions


if __name__ == "__main__":
    # Example usage
    from flatland.envs.rail_env import RailEnv

    # Assuming you have a RailEnv instance
    env: RailEnv = small_flatland_env()
    _ = env.reset()
    graph_builder = MultiDiGraphBuilder(env)