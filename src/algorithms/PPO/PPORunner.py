import torch
from torch import Tensor
from collections import defaultdict
from typing import Dict, Tuple, List, DefaultDict, Union

from src.utils.obs_utils import tree_observation_dict
from src.algorithms.PPO.PPOController import PPOController
from src.algorithms.PPO.PPORollout import PPORollout, PPOTransition

from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState

class PPORunner():
    def __init__(self, runner_handle: Union[int, str], env: RailEnv, controller: PPOController) -> None:
        self.env: RailEnv = env
        self.controller: PPOController = controller 
        self.rollout: DefaultDict[int, PPORollout] = defaultdict(PPORollout)


    def run(self, max_steps: int) -> Tuple[DefaultDict[int, PPORollout], Dict]:
        """
        Run a single episode in the environment and collect rollouts.

        Parameters:
            - env           RailEnv         The environment to run the episode in
            - controller    PPOController   The controller to use for action selection

        Returns:
            - rollout       PPORollout      The collected rollouts
            - stats         Dict            Statistics about the episode (e.g., rewards, lengths)
        """
        state_dict_np, _ = self.env.reset()
        state_dict_tensor: Dict[int, Tensor] = tree_observation_dict(state_dict_np, self.env.agents) #! this isn't correct
        self.obs_tensor_size = state_dict_tensor[0].size()

        self.prev_valid_state: Dict[int, Tensor] = state_dict_tensor
        self.prev_valid_action: Dict[int, int] = {}
        self.prev_valid_action_log_prob: Dict[int, Tensor] = {} #? is this correct?
        self.prev_step: Tensor = torch.zeros(len(state_tensor), dtype=torch.float64)
        self.neighbours_states: Dict[int, List[Tensor]] = defaultdict(list)

        steps_done = 0

        while True: 
            action_dict, log_probs = self._select_actions()
            next_state_dict, reward, done, _ = self.env.step(action_dict) # TODO: check if wrapping is necessary for reward and info
            next_state_tensor: Dict[int, Tensor] = tree_observation_dict(next_state_dict, self.env.agents)
            self._save_transitions(state_tensor, action_dict, log_probs, next_state_tensor, reward, done, steps_done)

            state_tensor = next_state_tensor
            steps_done += 1

            if done['__all__'] or steps_done >= max_steps:
                break
            
        percent_done = sum([1 for agent in self.env.agents if agent.state == TrainState.DONE]) / self.env.number_of_agents
        return self.rollout, {'reward': self.env.global_reward,
                              'percent_done': percent_done,
                              'steps_done': steps_done}


    def _select_actions(self, state: Dict[int, Tensor]) -> Tuple[Dict, Dict, Dict]:
        """
        Select actions for all agents based on their current states.
        """
        valid_handles: List = []
        internal_state: Dict = {}

        for handle in self.env.agents:
            #! original code uses self.env.obs_builder.deadlock_checker.is_deadlocked(handle) --> not sure what the new implementation is
            if self.env.agents[handle].state in (TrainState.MOVING, TrainState.READY_TO_DEPART) \
                and not self.env.agents[handle].state in (TrainState.STOPPED, TrainState.MALFUNCTION, TrainState.DONE):
                valid_handles.append(handle)
                if handle in state:
                    internal_state[handle] = state[handle]
                else: 
                    internal_state[handle] = torch.tensor() #! original code uses self.env.obs_builder._get_internal(handle) --> not sure what the new implementation is
        
        for handle in state.keys():
            if self.env.agents[handle].state in (TrainState.MOVING, TrainState.READY_TO_DEPART) \
                and not self.env.agents[handle].state in (TrainState.STOPPED, TrainState.MALFUNCTION, TrainState.DONE): #! see above
                valid_handles.append(handle)        
                #! original code uses self.env.obs_builder.encountered(handle) to decide on neighbours
                for neighbour_handle in self.env.agents:
                    if neighbour_handle in internal_state:
                        self.neighbours_states[handle].append(internal_state[neighbour_handle])
                    else:
                        self.neighbours_states[handle].append(torch.zeros(self.obs_tensor_size)) 

        action_dict, log_probs = self.controller.sample_action(valid_handles, state, self.neighbours_states)                    
        return action_dict, log_probs


    def _save_transitions(self, state_dict: Dict[int, Tensor], action_dict: Dict, log_probs: Dict, next_state: Dict, reward: Dict, done: Dict, step: int) -> None:
        """
        Save transitions for each agent in the environment.
        """
        self.prev_valid_state.update(state_dict)
        self.prev_valid_action.update(action_dict)
        self.prev_valid_action_log_prob.update(log_probs)
        for handle in state_dict.keys():
            self.prev_step[handle] = step

        for handle in next_state.keys():
            if not handle in self.prev_valid_state:
                # train just departed
                continue
            self.rollout[handle].append_transition(
                PPOTransition(self.prev_valid_state[handle],
                              self.prev_valid_action[handle],
                              self.prev_valid_action_log_prob[handle],
                              next_state[handle],
                              reward[handle],
                              done[handle],
                              torch.stack(self.neighbours_states[handle]),)
            )


    def _wrap(self, obs: Dict) -> Dict:
        for key, value in obs.items():
            if isinstance(value, Tensor):
                obs[key] = torch.tensor(value, dtype=torch.float64)
        return obs

