''' Contains the observation pre-processing functions for flatland TreeObs '''

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from utils.utils import max_lowerthan, min_greaterthan
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Union
from flatland.envs.observations import Node

def _split_node_into_feature_groups(node: Node) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    '''
    Splits the features of a single node into three arrays: 
        - data: the observation data in the form of distance to targets, switches, other agents, etc.
        - distance: the distance to the target
        - agent_data: the number of agents in the same direction, opposite direction, malfunctioning, and the minimum speed fraction of the slowest agent in the same direction
    '''
    data: np.ndarray = np.zeros(6)
    distance: np.ndarray = np.zeros(1)
    agent_data: np.ndarray = np.zeros(4) 

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
    #* missing agents ready to depart feature
    # agent_data[4] = node.num_agents_ready_to_depart

    return data, distance, agent_data


def _split_subtree_into_feature_groups(node: Node, current_tree_depth: int, max_tree_depth: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # return lists of -inf if the current node is -inf
    if node == -np.inf:
        remaining_depth: int = max_tree_depth - current_tree_depth
        #* this gives the right answer, but I don't understand how it is deduced
        num_remaining_nodes: int = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
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


def split_tree_into_feature_groups(tree: Node, max_tree_depth: int):
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


def normalise_tree_observation(observation, tree_depth: int, observation_radius=0):
    # extract feature data
    data, distance, agent_data = split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalise_to_range=True)
    agent_data = norm_obs_clip(agent_data, fixed_radius=1)

    normalised_observation = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalised_observation


def translate_observation(observation: dict, obs_type: str, max_depth: int = 0, n_nodes: int = 0) -> Tensor:
    ''' Transforms observations from flatland RailEnv to torch tensors 
    Global observations are of shape (n_agents, env_width, env_height, 23)
    Tree observations are of shape (n_nodes, 12)
    '''
    if obs_type == 'global':
        return global_observation_tensor(observation)
    elif obs_type == 'tree':
        return tree_observation_tensor(observation, max_depth, n_nodes)


def global_observation_tensor(observation: dict) -> Tensor:
    '''
    Transforms global observations from flatland RailEnv to torch tensors. 
    Ouput: 
     - observation: a tensor of shape (n_agents, env_width, env_height, 23)
    '''
    obs = np.zeros((len(observation), observation[0][0].shape[0], observation[0][0].shape[1], 23))
    for agent_id in observation.keys():
        obs[agent_id] = np.concatenate((observation[agent_id][0], 
                                                observation[agent_id][1], 
                                                observation[agent_id][2]), axis=2)
    return torch.tensor(observation, dtype=torch.float32)

def tree_observation_tensor(observation: dict, max_depth: int, n_nodes: int) -> Tensor:
    '''
    Transforms observations from flatland RailEnv to a torch tensor with shape (n_agents, n_nodes, 12)
    '''
    agents_obs = np.ndarray(shape=(len(observation), n_nodes, 12))
    for agent in observation.keys():
        agents_obs[agent] = split_tree(observation[agent], max_depth)
    return torch.Tensor(agents_obs)


def split_tree(tree: Node, max_depth: int):
    ''' Splits the tree observation into an ndarray of features, initial splitting function '''
    features = split_features(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        child_features = split_subtree(tree.childs[direction], 1, max_depth)
        features = np.concatenate((features, child_features), axis=0)

    return features


def split_subtree(node: Union[Node, float], current_depth: int, max_depth: int):
    ''' Recursively splits the subtree observation into an ndarray of features '''
    # case: terminal node
    if node == -np.inf:
        remaining_depth: int = max_depth - current_depth
        #* this gives the right answer, but I don't understand how it is deduced
        num_remaining_nodes: int = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        node_features = [-np.inf] * 12 * num_remaining_nodes
        return np.reshape(node_features, newshape=(num_remaining_nodes, 12))
    
    features = split_features(node)

    # case: leaf node
    if not node.childs:
        return features
    
    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        child_features = split_subtree(node.childs[direction], current_depth + 1, max_depth)
        features = np.concatenate((features, child_features), axis=0)

    return features


def split_features(node: Node) -> np.ndarray:
    ''' Splits the features of a single node into an ndarray of shape (1, 12)'''
    features = np.zeros(12)
    features[0] = node.dist_own_target_encountered
    features[1] = node.dist_other_target_encountered
    features[2] = node.dist_other_agent_encountered
    features[3] = node.dist_potential_conflict
    features[4] = node.dist_unusable_switch
    features[5] = node.dist_to_next_branch

    features[6] = node.dist_min_to_target

    features[7] = node.num_agents_same_direction
    features[8] = node.num_agents_opposite_direction
    features[9] = node.num_agents_malfunctioning
    features[10] = node.speed_min_fractional
    features[11] = node.num_agents_ready_to_depart

    features = np.expand_dims(features, axis=0)

    return features


    
