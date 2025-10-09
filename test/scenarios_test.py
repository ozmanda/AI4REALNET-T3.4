import os
import unittest
from typing import List
import matplotlib.pyplot as plt

from src.environments.scenario_loader import load_scenario_from_json
from src.environments.env_small import small_flatland_env

scenarios: List[str] = ['simple_ordering',
                        'simple_avoidance',
                        'overtaking',
                        'complex_ordering',
                        'complex_avoidance']

class TestScenarioLoader(unittest.TestCase):
    def setUp(self):
        self.scenarios = scenarios

    def test_small_flatland_env(self):
        """ Test the small flatland environment loading. """
        env = small_flatland_env()
        self.assertIsNotNone(env, "Failed to load small flatland environment")
        env.reset()
        img = env.render()
        rail = env.rail
        self.assertIsNotNone(img, "Failed to render small flatland environment")
        plt.imshow(img)
        plt.title("Rendered Small Flatland Environment")
        plt.axis('off')
        plt.show()

    def test_single_scenario(self):
        """ Test the scenario loading for a single scenario. """
        print(os.getcwd())
        scenario = 'src/environments/simple_ordering.json'
        env = load_scenario_from_json(scenario)
        self.assertIsNotNone(env, f"Failed to load scenario: {scenario}")
        env.reset()
        img = env.render()
        self.assertIsNotNone(img, f"Failed to render scenario: {scenario}")
        plt.imshow(img)
        plt.title(f"Rendered scenario: {scenario}")
        plt.axis('off')
        plt.show()

    def test_render_scenarios(self):
        """ Test the scenario loading for one scenario. """
        for idx, scenario in enumerate(self.scenarios):
            with self.subTest(scenario=scenario):
                env = load_scenario_from_json(scenario)
                self.assertIsNotNone(env, f"Failed to load scenario: {scenario}")
                env.reset()
                img = env.render()  # Ensure the environment can be rendered
                self.assertIsNotNone(img, f"Failed to render scenario: {scenario}")
                if img is not None:
                    plt.imshow(img)
                    plt.title(f"Rendered scenario: {scenario}")
                    plt.axis('off')
                    plt.savefig(os.path.join("test", "renders", f"scenario_{idx}.png"), bbox_inches='tight')
                    plt.close()  # Close the plot to free memory
