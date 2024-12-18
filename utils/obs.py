''' Contains the observation pre-processing functions for flatland TreeObs '''

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from utils.utils import max_lowerthan, min_greaterthan

def _split_node_into_feature_groups(node):
    data = np.zeros(6)
    distance = np.zeros(1)
    agent_data = np.zeros(4)

    data[0] = node.dist_own_target_encountered
    data[1] = node.dist_other_target_encountered
    data[2] = node.dist_other_agent_encountered
    data[3] = node.dist_potential_conflict
    data[4] = node.dist_unusable_switch
    data[5] = node.dist_to_next_branch

    distance[0] = node.dist_min_to_target

    agent_data[0] = node.num_agents_same_direction
    agent_data[1] = node.num_agents_opposite_direction
    agent_data[2] = node.num_agents_malfunctioning
    agent_data[3] = node.speed_min_fractional

    return data, distance, agent_data


def _split_subtree_into_feature_groups(node, current_tree_depth: int, max_tree_depth: int):
    # return lists of -inf if the current node is -inf
    if node == -np.inf:
        remaining_depth = max_tree_depth - current_tree_depth
        num_remaining_nodes = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        return [-np.inf] * num_remaining_nodes * 6, [-np.inf] * num_remaining_nodes, [-np.inf] * num_remaining_nodes * 4
    
    # extract node data
    data, distance, agent_data = _split_node_into_feature_groups(node)

    # if this is a leaf node, return the data
    if not node.childs:
        return data, distance, agent_data

    # call recursively for child nodes
    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        child_data, child_distance, child_agent_data = _split_subtree_into_feature_groups(node.childs[direction], current_tree_depth + 1, max_tree_depth)
        data = np.concatenate((data, child_data))
        distance = np.concatenate((distance, child_distance))
        agent_data = np.concatenate((agent_data, child_agent_data))
        
    return data, distance, agent_data


def split_tree_into_feature_groups(tree, max_tree_depth: int):
    data, distance, agent_data = _split_node_into_feature_groups(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        sub_data, sub_distance, sub_agent_data = _split_subtree_into_feature_groups(tree.childs[direction], 1, max_tree_depth)
        data = np.concatenate((data, sub_data))
        distance = np.concatenate((distance, sub_distance))
        agent_data = np.concatenate((agent_data, sub_agent_data))

    return data, distance, agent_data


def norm_obs_clip(observation, clip_min=-1, clip_max=1, fixed_radius=0, normalise_to_range=0):
    """
    Observation: expected to be a numerical array or list
    clip_min / clip_max: the range of the clipped observation, normally [-1, 1]
    fixed_radius: if > 0, it overrides the max value for normalisation, making it a fixed scaling factor
    normalise_to_range: if != 0, the normalisation is adjusted to start from the smallest positive value in the observation range rather than 0
    """
    if fixed_radius > 0:
        max_obs = fixed_radius
    else:
        # maximum value that is less than 1000
        max_obs = max(1, max_lowerthan(observation, 1000)) + 1

    min_obs = 0

    # normalise to the smallest positive value rather than 0
    if normalise_to_range:
        min_obs = min_greaterthan(observation, 0)
    
    # Edge case 1: max_obs is negative, set min_obs = max_obs to avoid zero or negative normalisation
    if min_obs >= max_obs:
        min_obs = max_obs
    # Edge case 2: 
    if max_obs == min_obs:
        return np.clip(np.array(observation) / max_obs, clip_min, clip_max)
    
    norm = np.abs(max_obs - min_obs)
    return np.clip((np.array(observation) - min_obs) / norm, clip_min, clip_max)


def normalise_observation(observation, tree_depth: int, observation_radius=0):
    # extract feature data
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalise_to_range=True)
    agent_data = norm_obs_clip(agent_data, fixed_radius=1)

    normalised_observation = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalised_observation