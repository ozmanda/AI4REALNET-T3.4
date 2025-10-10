import numpy as np
from typing import Dict
from utils.flatland_railway_extension.RailroadSwitchAnalyser import RailroadSwitchAnalyser
from flatland.envs.rail_env import RailEnv


def add_topological_data(env: RailEnv) -> RailEnv:
    """
    Add track data to the environment.

    :param env: The RailEnv instance.
    """

    analyser = RailroadSwitchAnalyser(env)
    switch_clusters: Dict[str] = analyser.railroad_switch_clusters()
    rail_clusters: Dict = analyser.connecting_edge_clusters()

    grid_height = env.height
    grid_width = env.width

    # Add switch clusters to the environment
    env.switch_grid = np.zeros((grid_height, grid_width), dtype=int)
    for cluster_id, cluster in switch_clusters.items():
        for resource in cluster:
            env.switch_grid[resource[0], resource[1]] = cluster_id

    env.rail_grid = np.zeros((grid_height, grid_width), dtype=int)
    for cluster_id, cluster in rail_clusters.items():
        for resource in cluster:
            env.rail_grid[resource[0], resource[1]] = cluster_id

    return env