import torch
from torch import Tensor
from collections import defaultdict
from typing import Dict, Tuple, List, DefaultDict, Union

from src.utils.observation.obs_utils import tree_observation_dict
from src.algorithms.PPO_JBR_HSE.PPOController import PPOController
from src.algorithms.PPO_JBR_HSE.PPORollout import PPORollout, PPOTransition

from src.configs.EnvConfig import FlatlandEnvConfig
from flatland.envs.rail_env import RailEnv
from flatland.envs.step_utils.states import TrainState


class PPORunner():
    def __init__(self, runner_handle: Union[int, str], env_config: FlatlandEnvConfig, controller: PPOController) -> None:
        self.env: RailEnv = env_config.create_env()
        self.controller: PPOController = controller 
        self.rollout: DefaultDict[int, PPORollout] = defaultdict(PPORollout)
        self.max_depth = self.env.obs_builder.max_depth

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
        state_array, info = self.env.reset()
        state_tensor: Dict[int, Tensor] = tree_observation_dict(state_array, self.max_depth) # TODO: fix this!
        self.obs_tensor_size = state_tensor[0].size()

        self.prev_valid_state: Dict[int, Tensor] = state_tensor
        self.prev_valid_action: Dict[int, int] = {}
        self.prev_valid_action_log_prob: Dict[int, Tensor] = {} #? is this correct?
        self.prev_step: Tensor = torch.zeros(len(state_tensor), dtype=torch.float64)
        self.neighbour_states: Dict[int, List[Tensor]] = defaultdict(list)

        steps_done = 0

        while True: 
            if any(info['action_required'].values()):
                action_dict, log_probs = self._select_actions(state_tensor)
                next_state_array, reward, done, info = self.env.step(action_dict)
            else:
                action_dict = {handle: 0 for handle in state_tensor.keys()}
                log_probs = {handle: torch.zeros(1, dtype=torch.float64) for handle in state_tensor.keys()}
                next_state_array, reward, done, info = self.env.step(action_dict) # TODO: check if wrapping is necessary for reward and info
                
            next_state_tensor: Dict[int, Tensor] = tree_observation_dict(next_state_array, self.max_depth)
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
        valid_agent_states: Dict = {}
        self.neighbour_states.clear()  # Reset neighbour states for the new step

        for agent in self.env.agents:
            if agent.state in (TrainState.MOVING, TrainState.READY_TO_DEPART) \
                and not agent.handle in self.env.motionCheck.svDeadlocked: 
                if agent.handle in state:
                    valid_agent_states[agent.handle] = state[agent.handle]
                else: 
                    valid_agent_states[agent.handle] = torch.zeros(self.controller.state_size) #? if there's no observation, why fill with a zero tensor?

        for handle in state.keys():
            if self.env.agents[handle].state in (TrainState.MOVING, TrainState.READY_TO_DEPART) \
                and not self.env.agents[handle].state in self.env.motionCheck.svDeadlocked:
                valid_handles.append(handle)        
                #! original code uses self.env.obs_builder.encountered(handle) to decide on neighbours
                for neighbour_agent in self.env.agents:
                    if neighbour_agent.handle in valid_agent_states.keys():
                        self.neighbour_states[handle].append(valid_agent_states[neighbour_agent.handle])
                    else:
                        self.neighbour_states[handle].append(torch.zeros(self.obs_tensor_size)) 
        if valid_handles:
            action_dict, log_probs = self.controller.sample_action(valid_handles, state, self.neighbour_states)              
            return action_dict, log_probs
        else: return {}, {}


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
                              torch.stack(self.neighbour_states[handle]) if self.neighbour_states[handle] else None,
                              )
            )


    def _wrap(self, obs: Dict) -> Dict:
        for key, value in obs.items():
            if isinstance(value, Tensor):
                obs[key] = torch.tensor(value, dtype=torch.float64)
        return obs

