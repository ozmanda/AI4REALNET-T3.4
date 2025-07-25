import os
import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, DefaultDict
from collections import defaultdict

from flatland.envs.rail_env import RailEnv
from flatland.envs.rail_env_action import RailEnvActions
from flatland.core.grid.grid4_utils import get_new_position
from flatland.envs.fast_methods import fast_position_equal, fast_argmax, fast_count_nonzero

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from src.environments.env_small import small_flatland_env
from flatland.envs.rail_env_action import RailEnvActions

n_directions: int = 4  # range [1:infty)
rail_actions: Dict[int, RailEnvActions] = {
    0: "North",
    1: "East",
    2: "South",
    3: "West",
}

class MultiDiGraphBuilder:
    def __init__(self, env: RailEnv):
        self.graph: nx.MultiDiGraph = nx.MultiDiGraph()
        self.env: RailEnv = env
        self._init_parameters()
        self._init_switch_analyser()
        self._generate_graph()

    def _init_parameters(self):
        self.height: int = self.env.height
        self.width: int = self.env.width
        self.max_depth: int = self.height * self.width * 10
        self.from_directions: List[int] = [i for i in range(n_directions)]

        self.nodes: Dict[str, Tuple[int, int]] = {}

    def _init_switch_analyser(self):
        """Initialize the switch analyser if needed."""
        self.switch_analyser: RailroadSwitchAnalyser = RailroadSwitchAnalyser(
            self.env,
            handle_diamond_crossing_as_a_switch=True,
            handle_dead_end_as_a_switch=True
        )
        pass

    def _generate_graph(self) -> None:
        start_node, start_direction = None, None
        for x, y in ((x, y) for x in range(self.height) for y in range(self.width)):
            _, nonzero_directions = self._get_valid_transitions((x, y)) 
            for direction in nonzero_directions:
                start_node, _, dead_end = self._find_next_node((x, y), direction)
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
        in_direction: int = (travel_direction + (n_directions // 2)) % n_directions
        self._current_depth += 1
        if self._current_depth > self.max_depth:
            raise ValueError("Maximum depth exceeded while traversing the graph.")
        
        # Get valid transitions from the current node
        _, nonzero_directions = self._get_valid_transitions(node)
        for travel_direction in nonzero_directions:
            if travel_direction == in_direction: # prevents going back to the previous node
                continue
            next_node, out_direction, dead_end = self._find_next_node(node, travel_direction)
            if next_node:
                self._add_edge(node, next_node, key=travel_direction) #! do we need to add the direction as a key?
                if f'{next_node[0]}_{next_node[1]}' not in self.nodes.keys():
                    self._add_node(next_node, dead_end=dead_end)
                    if not dead_end:
                        self._node_splitter(next_node, travel_direction=out_direction)


    def _find_next_node(self, previous_position: Tuple[int, int], travel_direction: int) -> Tuple[int, int]:
        """
        Find the next node in the graph based on the current position and direction of traversal.
        """
        # first transition to new cell
        n_transitions = 2
        depth = 0
        current_position = get_new_position(previous_position, travel_direction)
        
        while n_transitions == 2:
            valid_transitions, _ = self._get_valid_transitions(current_position)
            transitions: Tuple = self.env.rail.get_transitions(*current_position, travel_direction)
            n_transitions: int = np.sum(valid_transitions)

            if n_transitions == 0:
                break
            elif n_transitions > 2: # multiple transitions possible = switch
                return current_position, travel_direction, False
            elif n_transitions == 1:  # only one transition possible = dead end
                return current_position, travel_direction, True
            else:
                # continue in the new travel direction
                travel_direction = fast_argmax(transitions)
                current_position: Tuple = get_new_position(current_position, travel_direction)
                depth += 1

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


    def _add_edge(self, u: Tuple[int, int], v: Tuple[int, int], key=None, **attr):
        """Add an edge to the graph."""
        self.graph.add_edge(u, v, key=key, **attr)
        self.graph.add_edge(v, u, key=key, **attr)  # Add reverse edge for undirected graph behavior

    def get_graph(self):
        """Return the constructed graph."""
        return self.graph

    def clear(self):
        """Clear the graph."""
        self.graph.clear()

    def render(self):
        """Render the graph showing nodes and edges."""
        positions = nx.get_node_attributes(self.graph, 'position')
        dead_end_status = nx.get_node_attributes(self.graph, 'dead_end')
        node_colours = ['lightgray' if dead_end else 'lightblue' for dead_end in dead_end_status.values()]

        plt.figure()
        nx.draw(self.graph, 
                pos=positions,
                labels={node: node for node in self.graph.nodes()},
                node_color=node_colours,

                )
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
        plt.axis('off')
        plt.savefig(os.path.join("test", "renders", "graph_render.png"), bbox_inches='tight')
        plt.show()
        plt.close()


if __name__ == "__main__":
    # Example usage
    from flatland.envs.rail_env import RailEnv

    # Assuming you have a RailEnv instance
    env: RailEnv = small_flatland_env()
    _ = env.reset()
    graph_builder = MultiDiGraphBuilder(env)