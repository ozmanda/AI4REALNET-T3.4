'''
Adapted from IC3net action_utils.py
'''
from argparse import Namespace
import torch
from torch import Tensor
import numpy as np

def sample_action(action_probs: Tensor):
    '''
    Converts log-probabilities into probabilities and samples from each of the distributions. 
    '''
    log_p_a = action_probs
    p_a = [[z.exp() for z in x] for x in log_p_a]
    ret = torch.stack([torch.stack([torch.multinomial(x, 1).detach() for x in p]) for p in p_a])
    return ret


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