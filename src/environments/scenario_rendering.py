import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
from src.environments.scenario_loader import load_scenario_from_json

SCENARIO_DIR = Path(__file__).resolve().parent

scenarios: list[str] = ['simple_ordering',
                        'simple_avoidance',
                        'overtaking',
                        'complex_ordering',
                        'complex_avoidance',
                        'graph_test_connected_schedules']

renders_dir = SCENARIO_DIR / 'renders'
renders_dir.mkdir(parents=True, exist_ok=True)

for idx, scenario in enumerate(scenarios):
    scenario_path = SCENARIO_DIR / f'{scenario}.json'
    env = load_scenario_from_json(str(scenario_path))
    env.reset()
    img = env.render()  # Ensure the environment can be rendered
    if img is not None:
        plt.imshow(img)
        plt.title(f"Scenario {scenario}")
        plt.axis('off')
        output_path = renders_dir / f'scenario_{scenario}.png'
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()  # Close the plot to free memory
