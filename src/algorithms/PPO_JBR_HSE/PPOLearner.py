import ray
import torch
from torch import Tensor
from itertools import chain
from typing import Dict, Tuple, List
from torch.distributions import Categorical

from src.utils.tensor_utils import _permute_tensors
from src.utils.obs_utils import calculate_state_size
from src.training.loss import value_loss, value_loss_with_IS, policy_loss

from src.algorithms.PPO_JBR_HSE.PPOController import PPOController
from src.algorithms.PPO_JBR_HSE.PPORollout import PPORollout
from src.algorithms.PPO_JBR_HSE.PPORunner import PPORunner

from src.configs.ControllerConfigs import PPOControllerConfig
from src.configs.EnvConfig import FlatlandEnvConfig
from src.configs.OptimiserConfig import AdamConfig

class PPOLearner():
    """
    Learner class for the PPO algorithm, updates the policy based on experiences (rollouts).
    """
    def __init__(self,  controller_config: PPOControllerConfig, env_config: FlatlandEnvConfig, device: str, n_workers: int = 1) -> None: 
        self.n_workers = n_workers
        self.device = device

        n_nodes, state_size = calculate_state_size(env_config.observation_builder_config['max_depth'])
        controller_config.config_dict['n_nodes'] = n_nodes
        controller_config.config_dict['state_size'] = state_size

        self.max_depth = env_config['max_depth']
        (env_config['observation_builder_config']['max_depth'])
        controller_config['actor_config']['n_nodes'] = n_nodes # TODO: add n_nodes as a model parameter
        controller_config['critic_config']['n_nodes'] = n_nodes
        controller_config['state_size'] = state_size
        self.controller: PPOController = controller_config.create_controller()
        # self.judge = 
        self.device = device

        self.training_iterations = 2000
        self.exploit_iterations = 500

        num_gpus = 0
        if device == torch.device('cuda'):
            num_gpus = 1

        ray.init(num_gpus=num_gpus)

        self.workers = [None] * n_workers
        for runner_handle in range(n_workers):
            self.workers[runner_handle] = PPORunner(runner_handle, env_config, num_gpus) # TODO: check this initialisation -> environment must be passed
            env_config.update_random_seed()

        self.batch_size = controller_config.batch_size
        self.gae_horizon = controller_config.gae_horizon
        self.value_loss_coefficient = controller_config.value_loss_coefficient
        self.entropy_coefficient = controller_config.entropy_coefficient
        self.lam = controller_config.lam
        self.gamma = controller_config.gamma
        self.epochs_update = controller_config.n_epochs_update
        self.clip_epsilon = controller_config.clip_epsilon

        optimiser_config: AdamConfig = controller_config.optimiser_config
        self.optimiser = optimiser_config.create_optimizer(chain(self.controller.critic_network.parameters(), self.controller.actor_network.parameters()))


    def _optimise(self, rollouts_dict: Dict[int, PPORollout]):
        # all agent rollouts are combined
        rollouts = [rollout for rollout in rollouts_dict.values() if not rollout.is_empty()]

        if not rollouts:
            return

        combined_rollout = PPORollout.combine_rollouts(rollouts)
        states, actions, log_probs, rewards, next_states, dones, neighbour_states, steps = combined_rollout.unzip_ppo_transitions()
        gae = combined_rollout.gae.to(self.device)

        for _ in range(self.epochs_update): 
            states, actions, log_probs, rewards, next_states, dones, neighbour_states, steps = _permute_tensors([states, actions, log_probs, rewards, next_states, dones, neighbour_states, steps])

            # iterate through the rollout in batches
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                loss = self._loss(states[start_idx:end_idx], actions[start_idx:end_idx], log_probs[start_idx:end_idx], 
                                  rewards[start_idx:end_idx], next_states[start_idx:end_idx], dones[start_idx:end_idx], 
                                  gae[start_idx:end_idx], neighbour_states[start_idx:end_idx], steps[start_idx:end_idx])
                
                self.optimiser.zero_grad()
                loss.backward()
                self.optimiser.step()

        self.controller.actor_soft_update(tau=0.05)


    def _loss(self, states: Tensor, actions: int, old_log_probs, rewards, next_states, dones, gaes, neighbours_states, steps) -> float:
        state_values = self.controller.critic_network(states)

        with torch.no_grad():
            next_state_values = self.controller.critic_network(next_states)
        
        logits = self.controller._make_logits(states, neighbours_states)

        action_distribution = Categorical(logits=logits)
        new_log_probs = action_distribution.log_prob(actions)

        critic_loss = value_loss_with_IS(state_values, next_state_values, new_log_probs, old_log_probs, rewards, dones, self.gamma, steps)
        actor_loss = policy_loss(gaes, new_log_probs, old_log_probs, self.clip_epsilon)
        entropy_loss = -action_distribution.entropy().mean()

        return critic_loss*self.value_loss_coefficient + actor_loss - entropy_loss*self.entropy_coefficient


    def rollouts(self, max_optim_steps: int, max_episodes:int) -> None:
        """
        Run the PPO algorithm for a given number of episodes and optimisation steps.

        Parameters:
            - max_optim_steps   int     Maximum number of optimisation steps per episode
            - max_episodes      int     Maximum number of episodes to run
        """
        # TODO: add logging
        controller_parameters = self.controller.get_network_parameters(device=torch.device('cpu'))
        rollouts_list = [worker.run.remote(controller_parameters, max_optim_steps, max_episodes) for worker in self.workers]
        current_steps, current_epsiodes = 0, 0

        while True:
            done_id, rollouts_list = ray.wait(rollouts_list)
            rollout, info = ray.get(done_id)[0]

            current_steps += info['steps_done']
            current_epsiodes += 1

            print(current_epsiodes, info['reward'])

            if current_epsiodes % 100 == 0:
                continue #TODO: save logs
            if current_epsiodes % 250 == 0:
                self.controller.save_controller(f'./models/ppo_{current_epsiodes}/controller.torch')
                # TODO: judge?
            if (current_steps % (self.training_iterations + self.exploit_iterations)) < self.training_iterations:
                # TODO: judge?
                self._optimise(rollout)

            if current_steps >= max_optim_steps or current_epsiodes >= max_episodes:
                break
            
            controller_parameters = self.controller.get_network_parameters(device=torch.device('cpu'))
            rollouts_list.extend([self.workers[info['handle']].run.remote(controller_parameters)])
        
        # TODO: logging