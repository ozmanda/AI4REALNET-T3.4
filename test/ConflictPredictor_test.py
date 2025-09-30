
import os

import unittest
from src.environments.scenario_loader import load_scenario_from_json
from src.utils.graph.ConflictPredictor import ConflictPredictor
from src.utils.graph.MultiDiGraphBuilder import MultiDiGraphBuilder
from src.algorithms.Base.RandomController import RandomController

class TestConflictPredictor(unittest.TestCase):
    def setUp(self):
        scenario = 'simple_avoidance'
        scenario_dir = os.path.join(os.getcwd(), 'src', 'environments', f'{scenario}.json')
        self.env = load_scenario_from_json(scenario_dir)
        self.env.reset()
        self.controller = RandomController(n_actions=self.env.action_space[0], n_agents=self.env.number_of_agents)
        self.conflict_predictor = ConflictPredictor(self.env)
                                                    
    def test_agent_placement(self):
        self.conflict_predictor._place_agents()
        pass

    def test_conflict_detection(self):
        """ Test if conflicts are detected correctly in the environment. """    

        self.assertIsNotNone(self.conflict_predictor.graph, "Graph should be generated successfully.")