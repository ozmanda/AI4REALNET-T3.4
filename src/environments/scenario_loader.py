import os
import json
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.timetable_utils import Line, Timetable
from flatland.core.transition_map import GridTransitionMap
from flatland.core.grid.rail_env_grid import RailEnvTransitions


def rail_generator_from_grid_map(grid_map, level_free_positions):
    def rail_generator(*args, **kwargs):
        return grid_map, {
            "agents_hints": {"city_positions": {}},
            "level_free_positions": level_free_positions,
        }

    return rail_generator

def line_generator_from_line(line):
    def line_generator(*args, **kwargs):
        return line

    return line_generator


def timetable_generator_from_timetable(timetable):
    def timetable_generator(*args, **kwargs):
        return timetable

    return timetable_generator


def load_scenario_from_json(scenario_name):
    dir = os.path.join(os.getcwd(), 'src', 'utils', 'environments', f'{scenario_name}.json')
    with open(dir, 'r') as f:
        data = json.load(f)

    width = data['gridDimensions']['cols']
    height = data['gridDimensions']['rows']
    number_of_agents = len(data['flatland line']['agent_positions'])

    line = Line(
        agent_positions = [[tuple(coords) for coords in positions] for positions in data['flatland line']['agent_positions']],
        agent_directions = data['flatland line']['agent_directions'],
        agent_targets = [tuple(coords) for coords in data['flatland line']['agent_targets']],
        agent_speeds = data['flatland line']['agent_speeds'],
    )

    timetable = Timetable(
        earliest_departures = data['flatland timetable']['earliest_departures'],
        latest_arrivals = data['flatland timetable']['latest_arrivals'],
        max_episode_steps = data['flatland timetable']['max_episode_steps']
    )

    grid = GridTransitionMap(width=width, height=height, transitions=RailEnvTransitions())
    grid.grid = np.array(data['grid'])

    level_free_positions = [tuple(item) for item in data['overpasses']]


    env = RailEnv(
        width = width,
        height = height,
        number_of_agents = number_of_agents,
        rail_generator = rail_generator_from_grid_map(grid, level_free_positions),
        line_generator = line_generator_from_line(line),
        timetable_generator = timetable_generator_from_timetable(timetable),
    )

    return env