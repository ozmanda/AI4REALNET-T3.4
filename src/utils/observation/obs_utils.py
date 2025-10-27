''' Contains the observation pre-processing functions for flatland TreeObs '''

import numpy as np
from flatland.envs.observations import TreeObsForRailEnv
from src.utils.utils import max_lowerthan, min_greaterthan
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Union, Dict, List
from flatland.envs.observations import Node

def calculate_state_size(max_depth: int) -> Tuple[int, int]:
    '''
    Calculates the number of nodes in the tree and the state size of the tree observation based on the maximum depth of the tree.
    
    Returns: 
      - n_nodes: the number of nodes in the tree
      - state_size: the size of the state space, which is the number of nodes multiplied by 12
    '''
    n_nodes = _calculate_tree_nodes(max_depth)
    return n_nodes, n_nodes * 12

def _calculate_tree_nodes(max_depth: int) -> int:   
    return int((4 ** (max_depth + 1) - 1) / 3) #* geometric progression

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


def _split_tree_into_feature_groups(tree: Node, max_tree_depth: int):
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
    data, distance, agent_data = _split_tree_into_feature_groups(observation, tree_depth)

    data = norm_obs_clip(data, fixed_radius=observation_radius)
    distance = norm_obs_clip(distance, normalise_to_range=True)
    agent_data = norm_obs_clip(agent_data, fixed_radius=1)

    normalised_observation = np.concatenate((np.concatenate((data, distance)), agent_data))
    return normalised_observation


def obs_dict_to_tensor(observation: Dict[int, Node], obs_type: str, n_agents: int, max_depth: int, n_nodes: int) -> Tensor:
    ''' 
    Transforms observations from flatland RailEnv to torch tensors, also flattening them to be
    two-dimensional. See https://flatland.aicrowd.com/environment/observations.html for more 
    information on observation types and structure.

    For global observations:    (n_agents, env_width * env_height * 23)
    Tree observations:          (n_agents, n_nodes * 12)
    '''
    n_agents = len(observation)
    if obs_type == 'global':
        obs_tensor = global_observation_tensor(observation)
    elif obs_type == 'tree':
        obs_tensor = tree_observation_tensor(observation, max_depth, n_nodes)

    # flatland fills missing values with -inf - replace with zero passing
    obs_tensor[obs_tensor == -np.inf] = 0
    obs_tensor[obs_tensor == np.inf] = 0

    return obs_tensor.view(n_agents, -1)


def global_observation_tensor(observation: Dict[int, np.ndarray]) -> Tensor:
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


def tree_observation_dict(observation: dict, max_depth: int) -> Dict[int, Tensor]:
    agent_obs: Dict[int, Tensor] = {}
    for agent in observation.keys():
        obs = split_tree(observation[agent], max_depth).reshape(-1)
        obs[obs == -np.inf] = -1  
        obs[obs == np.inf] = -1
        agent_obs[agent] = obs
    return agent_obs


def split_tree(tree: Node, max_depth: int) -> Tensor:
    ''' Splits the tree observation into an ndarray of features, initial splitting function '''
    features: Tensor = split_features(tree)

    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        child_features: Tensor = split_subtree(tree.childs[direction], 1, max_depth)
        features: Tensor = torch.cat((features, child_features), axis=0)

    return features


def split_subtree(node: Union[Node, float], current_depth: int, max_depth: int) -> Tensor:
    ''' Recursively splits the subtree observation into an ndarray of features '''
    # case: terminal node
    if node == -np.inf or node == np.inf:
        remaining_depth: int = max_depth - current_depth
        #* this gives the right answer, but I don't understand how it is deduced
        num_remaining_nodes: int = int((4 ** (remaining_depth + 1) - 1) / (4 - 1))
        node_features = [-1] * 12 * num_remaining_nodes #! this is changed from -np.inf to -1 to avoid issues with torch
        node_features = torch.tensor(node_features, dtype=torch.float32).reshape(num_remaining_nodes, 12)
        return node_features
    
    features: Tensor = split_features(node)

    # case: leaf node
    if not node.childs:
        return features
    
    for direction in TreeObsForRailEnv.tree_explored_actions_char:
        child_features: Tensor = split_subtree(node.childs[direction], current_depth + 1, max_depth)
        features: Tensor = torch.cat((features, child_features), axis=0)

    return features


def split_features(node: Node) -> Tensor:
    ''' Splits the features of a single node into a Tensor of shape (1, 12)'''
    features = torch.zeros(12)
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

    features = features.unsqueeze(0)
    return features


def direction_tensor(neighbour_depth) -> List[Tensor]:
    """
    Adapted from the JBR_HSE Solution to the Flatland 2020 Challenge: https://github.com/jbr-ai-labs/NeurIPS2020-Flatland-Competition-Solution/tree/master
    Adjusted to consider a ternary tree, which is the form of the flatland tree observation (left, straight, right)

    Each depth begins at left_child_index and ends at right_child_index, for example depth 1 = [0, 2], depth 2 = [3, 11], etc.
    """
    # TODO: adjust to quartenary tree
    direction_tensor: Tensor = torch.zeros((2 ** (neighbour_depth + 1) - 2, neighbour_depth), dtype = torch.float64)
    for depth in range(1, neighbour_depth + 1):
        left_parent_index: int = 2 ** (depth - 1) -2
        right_parent_index: int = 2 ** depth - 2
        
        left_child_index: int = 2 ** depth - 2
        right_child_index: int = 2 ** (depth + 1) - 2

        if depth > 1:
            direction_tensor[left_child_index : right_child_index : 2] = direction_tensor[left_parent_index : right_parent_index]
            direction_tensor[left_child_index+1 : right_child_index : 2] = direction_tensor[left_parent_index : right_parent_index]
        
        direction_tensor[left_child_index : right_child_index : 2, depth-1] = -1
        direction_tensor[left_child_index+1 : right_child_index : 2, depth-1] = 1

    return direction_tensor


def get_depth(size, n_features = 12) -> int:
    """
    Calculates the depth of a tree based on the size of the observation and the number of features per node. Default is for the flatland environment tree observation with 12 features per node.
    """
    for i in range(10):
        if ((4 ** (i+1) - 1) // (4 - 1)) * n_features == size:
            return i
    raise ValueError('Depth outside of range')
        