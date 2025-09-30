import heapq
import itertools

import numpy as np
import networkx as nx

from itertools import islice, product
from typing import Tuple, Dict, List, Union, Any
from flatland.envs.agent_utils import EnvAgent

class PathGenerator:
    def __init__(self, graph: nx.MultiDiGraph, station_lookup: Dict[Tuple[int, int], Union[str, int]], k: int = 4, weight: str = 'length', weight_goal: str = 'min'):
        self.k = k
        self.graph = graph
        self.weight = weight
        self.station_lookup = station_lookup
        self._init_digraph(weight_goal)

    def _init_digraph(self, weight_goal: str): 
        self.digraph = nx.DiGraph()
        for u, v, data in self.graph.edges(data=True):
            weight = data['attr'].get(self.weight)
            if self.digraph.has_edge(u, v):
                if weight_goal == 'min':
                    if self.digraph[u][v]['attr'].get(self.weight) > weight:
                        self.digraph[u][v]['attr'][self.weight] = weight

                elif weight_goal == 'max':
                    if self.digraph[u][v]['attr'].get(self.weight) < weight:
                        self.digraph[u][v]['attr'][self.weight] = weight

                else:
                    raise ValueError(f'Incompatible weight goals: {weight_goal}, choose either "max" or "min"')
            else:
                self.digraph.add_edge(u, v, **data)

    def get_k_shortest_paths(self):
        """
        Get the k-shortest paths between each every station-pair.
        """
        # TODO: more efficient way to save this data?
        station_nodes = self.station_lookup.keys()
        self.path_lookup: Dict[Tuple[Tuple[int, int], Tuple[int, int]], List[str]] = {}
        self.paths: Dict[str, List] = {}  # Dictionary of path IDs to paths
        for source_node in station_nodes:
            for target_node in station_nodes:
                if source_node != target_node:
                    for k, path in enumerate(self._get_k_shortest_paths(source_node, target_node, self.k, weight=self.weight)):
                        pathID = (source_node, target_node, k)
                        self.paths[pathID] = path
                        if (source_node, target_node) not in self.path_lookup.keys():
                            self.path_lookup[(source_node, target_node)] = [pathID]
                        else:
                            self.path_lookup[(source_node, target_node)].append(pathID)

        return self.path_lookup, self.paths


    def _get_k_shortest_paths(self, source_node: Tuple[int, int], target_node: Tuple[int, int], 
                              k: int, weight: int = None) -> List[List[Tuple[int, int, int]]]:
        """
        Find the k shortest paths in the graph - MultiDiGraph is not implemented yet, using DiGraph instead -> requires some helper
        functions to manage nodes connected by multiple edges. 

        :param source_node: The starting node.
        :param target_node: The target node.
        :param k: The number of shortest paths to find.
        :param weight: The name of the edge attribute to consider for the paths
        """
        path_generator = nx.shortest_simple_paths(self.digraph, source_node, target_node, weight=weight)
        k_shortest_paths = [path for path in path_generator][:k]
        k_shortest_edge_paths = self._multidigraph_correction(k_shortest_paths, weight=weight)
        return k_shortest_edge_paths


    def _multidigraph_correction(self, node_paths: List[List[Tuple[int, int]]], weight: str = None) -> List[List[Tuple[int, int, int]]]:
        """
        Translate DiGraph node paths to MultiDiGraph edge paths. 
        """
        heap = [] # max-heap of size k storing (-total_weight, edge_path)
        for path in node_paths:
            edge_options_per_step: List = []
            for u, v in zip(path[:-1], path[1:]):
                edges = [(u, v, key) for key in self.graph[u][v].keys()]
                edge_options_per_step.append(edges)
            
            for combination in itertools.product(*edge_options_per_step):
                total_weight = sum(self.graph[u][v][key]['attr'][weight] for u, v, key in combination)
    
                # push into heap
                if len(heap) < self.k:
                    heapq.heappush(heap, (-total_weight, combination))
                else:
                    # only keep if better than current worst
                    if -heap[0][0] > total_weight:
                        heapq.heapreplace(heap, (-total_weight, combination))

        # extract paths from heap
        k_shortest_edge_paths: List[List[Tuple[int, int, int]]] = [path for _, path in sorted(heap, key=lambda x: -x[0])]
        return k_shortest_edge_paths


    def get_conflict_matrix(self, paths: Dict[str, List]) -> np.ndarray:
        """
        Identify conflicts in the graph based on overlapping paths. Method: if any of the subpaths include the same node,
        a conflict is detected.
        """
        self.path_conflict_matrix = np.zeros((len(paths), len(paths)))
        for i, path in enumerate(paths):
            for j, other_path in enumerate(paths):
                if i != j and set(path) & set(other_path):
                    self.path_conflict_matrix[i, j] = 1
        return self.path_conflict_matrix
    

    def _edge_occupation_time(self, edge: Tuple[Tuple[int, int], Tuple[int, int], int], agent: EnvAgent):
        """
        Determine the time an agent occupies a specific edge.
        """
        pass

    def _node_occupation_time(self, node: Tuple[int, int], agent: EnvAgent):
        """
        Determine the time an agent occupies a specific node.
        """
        pass

