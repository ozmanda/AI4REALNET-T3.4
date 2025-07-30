import unittest
from typing import List, Dict, Tuple

from src.utils.flatland_railway_extension.MultiDiGraphBuilder import MultiDiGraphBuilder
from src.environments.scenario_loader import load_scenario_from_json
from src.environments.env_small import small_flatland_env

import os
from matplotlib import pyplot as plt


scenarios: List[str] = ['simple_avoidance',
                        'simple_ordering',
                        'overtaking', 
                        'complex_ordering']

complex_graph_test: str = 'graph_test_connected'

class TestMultiDiGraphBuilder_Complex(unittest.TestCase):
    def setUp(self): 
        scenario_dir = os.path.join(os.getcwd(), 'src', 'environments', f'{complex_graph_test}.json')
        self.env = load_scenario_from_json(scenario_dir)
        self.env.reset()

        self.graph_results: Dict[Tuple[int, int], int] = {
            (0,10): 1,
            (1,10): 3,
            (2, 9): 3,
            (3, 9): 3,
            (2, 5): 3,
            (3, 5): 4,
            (4, 6): 1,
            (5, 5): 4,
            (4, 8): 3,
            (3, 9): 3,
            (2,11): 3,
            (3,11): 3,
            (2,15): 3,
            (3,15): 4,
            (4,14): 1,
            (5,15): 4,
            (4,12): 3,
        }
        self.n_edges = sum(self.graph_results.values())
        self.n_nodes = len(self.graph_results)

        self.graph = MultiDiGraphBuilder(self.env)

    def test_graph_generation(self):
        """ Test the graph generation for one scenario. """
        self.env.reset()
        self.assertIsNotNone(self.graph.graph, "Graph should be generated successfully.")
        self.assertGreater(len(self.graph.graph.nodes), 0, "Graph should contain nodes.")
        self.assertGreater(len(self.graph.graph.edges), 0, "Graph should contain edges.")

    def test_node_generation(self):
        """ Test the node generation in the graph. """
        self.assertEqual(len(self.graph.graph.nodes), self.n_nodes, "Graph should contain nodes.")
        for node in self.graph_results.keys():
            self.assertIn(node, self.graph.graph.nodes, f"Node {node} should be present in the graph.")

    def test_edge_generation(self):
        """ Test the edge generation in the graph. """
        self.assertEqual(len(self.graph.graph.edges), self.n_edges, "Graph should contain edges.")
        for edge in self.graph_results.keys():
            self.assertEqual(len(self.graph.graph.edges(edge)), self.graph_results[edge],
                             f"Edge {edge} should have {self.graph_results[edge]} connections.")

    def test_graph_render(self):
        """ Test the graph generation for a complex scenario. """
        self.graph.render(os.path.join("test", "renders", f"graphbuilder_test_scenario_graph.png"))


    