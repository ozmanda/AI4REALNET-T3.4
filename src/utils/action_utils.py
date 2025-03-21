'''
Adapted from IC3net action_utils.py
'''
import torch
import numpy as np
from torch import Tensor
from typing import List, Dict
from argparse import Namespace

def sample_action(action_log_probs: Tensor) -> Tensor:
    '''
    Converts log-probabilities into probabilities and samples from the distribution. 
     -> action_probs: Tensor of length n_actions
    '''
    if action_log_probs.dim() == 3:
        action_log_probs = action_log_probs.view(-1, action_log_probs.size(-1))
    action_probs = action_log_probs.exp()
    sampled_action: Tensor = torch.multinomial(action_probs, 1).detach()
    return sampled_action


# TODO: check this
def action_tensor_to_dict(action: Tensor, agent_ids: List[int]) -> Dict[int, int]:
    ''' Converts a tensor of actions to a dictionary with agent_ids as keys '''
    return {agent_ids[i]: action[i] for i in range(len(agent_ids))}


def translate_discrete_action(args: Namespace, env, action: int): 
    '''
    Translates a sampled discrete action to fit the action space of the environment, copied directly from IC3Net action_utils.py, not adjusted yet (also not used)
    '''
    # TODO: adapt this if needed
    actual: np.ndarray = np.zeros(len(action))
    for i in range(len(action)):
        low = env.action_space.low[i]
        high = env.action_space.high[i]
        actual[i] = action[i].data.squeeze()[0] * (high - low) / (args.naction_heads[i] - 1) + low
    action = [x.squeeze().data[0] for x in action]
    return action, actual


def translate_continuous_action(args: Namespace, env, action: int): 
    '''
    Translates a sampled continuous action to fit the action space of the environment, copied directly from IC3Net action_utils.py, not adjusted yet (also not used)
    '''
    # TODO: adapt this if needed
    action = action.data[0].numpy()
    cp_action = action.copy()
    # clip and scale action to correct range
    for i in range(len(action)):
        low = env.action_space.low[i]
        high = env.action_space.high[i]
        cp_action[i] = cp_action[i] * args.action_scale
        cp_action[i] = max(-1.0, min(cp_action[i], 1.0))
        cp_action[i] = 0.5 * (cp_action[i] + 1.0) * (high - low) + low
    return action, cp_action