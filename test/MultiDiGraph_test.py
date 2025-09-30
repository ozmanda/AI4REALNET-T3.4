import os
import unittest
from typing import List, Dict, Tuple, Union

from src.utils.graph.MultiDiGraphBuilder import MultiDiGraphBuilder
from src.environments.scenario_loader import load_scenario_from_json

from flatland.envs.rail_env import RailEnv

import matplotlib.pyplot as plt

scenarios: List[str] = ['simple_avoidance',
                        'simple_ordering',
                        'overtaking', 
                        'complex_ordering']

complex_graph_test: str = 'graph_test_connected_schedules'

class TestMultiDiGraphBuilder_Complex(unittest.TestCase):
    def connected_graph_test(self): 
        scenario_dir = os.path.join(os.getcwd(), 'src', 'environments', 'graph_test_connected.json')
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

    def station_graph_test(self): 
        scenario_dir = os.path.join(os.getcwd(), 'src', 'environments', 'graph_test_connected_schedules.json')
        self.env = load_scenario_from_json(scenario_dir)
        self.env.reset()
        img = self.env.render()
        plt.imshow(img)
        plt.savefig(os.path.join('test', 'renders', 'test_stations'))
        self.graph = MultiDiGraphBuilder(self.env)
        
        # station locations (row, column)
        self.stations: Dict[Union[int, str], Tuple[int, int]] = {
            1: (8,5),
            2: (8,15),
            3: (5,10),
            4: (2,4),
            5: (2,16)
        }

        # node locations (row, column)
        self.graph_results: Dict[Tuple[int, int], int] = {
            (2, 5): 3,
            (2, 9): 3,
            (2,11): 3,
            (2,15): 3,
            (3, 5): 4,
            (3, 9): 3,
            (3,11): 3,
            (3,15): 4,
            (4, 6): 1,
            (4, 8): 3,
            (4,12): 3,
            (4,14): 1,
            (5, 5): 4,
            (5,15): 4,
        }

    def test_graph_generation(self):
        """ Test the graph generation for one scenario. """
        self.connected_graph_test()
        self.assertIsNotNone(self.graph.graph, "No graph generated.")
        self.assertGreater(len(self.graph.graph.nodes), 0, "Graph should contain nodes.")
        self.assertGreater(len(self.graph.graph.edges), 0, "Graph should contain edges.")

    def test_node_generation(self):
        """ Test the node generation in the graph. """
        self.connected_graph_test()
        self.assertEqual(len(self.graph.graph.nodes), self.n_nodes, "Graph should contain nodes.")
        for node in self.graph_results.keys():
            self.assertIn(node, self.graph.graph.nodes, f"Node {node} should be present in the graph.")

    def test_edge_generation(self):
        """ Test the edge generation in the graph. """
        self.connected_graph_test()
        self.assertEqual(len(self.graph.graph.edges), self.n_edges, "Graph should contain edges.")
        for edge in self.graph_results.keys():
            self.assertEqual(len(self.graph.graph.edges(edge)), self.graph_results[edge],
                             f"Edge {edge} should have {self.graph_results[edge]} connections.")

    def test_graph_render(self):
        """ Test the graph generation for a complex scenario. """
        self.connected_graph_test()
        self.graph.render(os.path.join("test", "renders", f"graphbuilder_test_scenario_graph.png"))


    def test_stations(self):
        """ Test the recognition of stations in the graph and identification of IDs """
        self.station_graph_test()
        for station_ID, coordinates in self.graph.stations_dict.items():
            self.assertIn(coordinates, self.graph.graph.nodes, f"Station {station_ID} should be present in the graph.")
            self.assertEqual(self.stations[int(station_ID)], coordinates,
                             f"Coordinates for station {station_ID} should be {coordinates}.")
            
    def test_path_generation(self):
        """ Test the generation of k-shortest paths for station pairs """
        self.station_graph_test()
        self.graph.station_path_data()
        path_pairs = [
            ((2,4),(8,5)),
            ((2,4),(5,10)),
            ((2,4),(2,16)),
            ((2,4),(8,15)),
            ((8,5),(5,10)),
            ((8,5),(2,16)),
            ((8,5),(8,15)),
            ((5,10),(2,16)),
            ((5,10),(8,15)),
            ((2,16),(8,15))
        ]
        for pair in path_pairs:
            self.assertIn(pair, self.graph.path_lookup.keys(), f"Path pair {pair} should be present in the path lookup.")


    def test_conflict_identification(self):
        """ Test the identification of conflicts between paths """
        agent_A = {
            "path": [((2, 4), (8, 5), 0)],
            "speed": 1.0,
            "departure_time": 0.0
        }
        agent_B = {
            "path": [((2, 4), (5, 10), 0)],
            "speed": 0.75,
            "departure_time": 0.0
        }
        self.assertTrue(self.graph.check_conflict(agent_A, agent_B), "Agents A and B should conflict.")