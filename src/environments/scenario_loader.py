import os
import json
import numpy as np

from flatland.envs.rail_env import RailEnv
from flatland.envs.timetable_utils import Line, Timetable
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid.rail_env_grid import RailEnvTransitions
from flatland.envs.rail_trainrun_data_structures import Waypoint
from flatland.envs.observations import GlobalObsForRailEnv


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


def load_scenario_from_json(scenario_path: str, observation_builder=GlobalObsForRailEnv()) -> RailEnv:
    with open(scenario_path, 'r') as f:
        data = json.load(f)

    width = data['gridDimensions']['cols']
    height = data['gridDimensions']['rows']
    number_of_agents = len(data['flatland line']['agent_positions'])

    line = Line(
        agent_waypoints=get_waypoints(data['flatland line']['agent_positions'], data['flatland line']['agent_targets'], data['flatland line']['agent_directions']),
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
        obs_builder_object = observation_builder,
    )

    env.stations = data['stations']

    return env


def get_waypoints(agent_positions, agent_targets, agent_directions):
    agent_waypoints = {i: [] for i in range(len(agent_positions))}
    for agent, wpts in enumerate(agent_positions):
        for idx, wpt in enumerate(wpts):
            agent_waypoints[agent].append([Waypoint(position=tuple(wpt), direction=agent_directions[agent][idx])])
    for agent, wpt in enumerate(agent_targets):
        agent_waypoints[agent].append([Waypoint(position=tuple(wpt), direction=agent_directions[agent][-1])])
    return agent_waypoints


def get_num_agents(scenario_path: str) -> int:
    with open(scenario_path, 'r') as f:
        data = json.load(f)
    return len(data['flatland line']['agent_positions'])