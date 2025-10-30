import os
import wandb
import numpy as np
from itertools import chain
from typing import List, Dict, Union, Tuple, Optional, Any

import torch
from torch import Tensor
import torch.optim as optim

from src.controllers.BaseController import Controller
from src.controllers.PPOController import PPOController
from src.controllers.LSTMController import LSTMController
from src.configs.ControllerConfigs import ControllerConfig
from src.configs.EnvConfig import FlatlandEnvConfig
from src.algorithms.loss import value_loss, value_loss_with_IS, policy_loss
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.utils.observation.obs_utils import obs_dict_to_tensor
from src.utils.observation.normalisation import FlatlandNormalisation


class PPOLearner():
    """
    Learner class for the PPO Algorithm.
    """
    def __init__(self, controller_config: ControllerConfig, learner_config: Dict, env_config: FlatlandEnvConfig, device: str = None) -> None:
        # Initialise environment and set controller / learning parameters
        self.env_config = env_config
        self._init_learning_params(learner_config)
        self._init_controller(controller_config)

        # Initialise environment
        self.obs_type: str = self.env_config.observation_builder_config['type']
        self.max_depth: int = self.env_config.observation_builder_config['max_depth']
        self.env = env_config.create_env()
        self._init_normalisation()

        # Initialise the optimiser
        self._build_optimiser(learner_config['optimiser_config'])
        self.epochs: int = 0
        self.total_episodes: int = 0

        # Initialise wandb for logging
        self._init_wandb(learner_config)

        # Track shutdown state so we can guarantee model persistence on exit
        self._shutdown_requested: bool = False
        self._shutdown_in_progress: bool = False
        self._register_signal_handlers()


    def _init_controller(self, config: ControllerConfig) -> None:
        self.controller_config = config
        self.n_nodes: int = config.config_dict['n_nodes']
        self.state_size: int = config.config_dict['state_size']
        self.controller: Union[PPOController, LSTMController, Controller] = config.create_controller()


    def _init_learning_params(self, learner_config: Dict) -> None:
        self.max_steps: int = learner_config['max_steps']
        self.max_steps_per_episode: int = learner_config['max_steps_per_episode']
        self.target_updates: int = learner_config['target_updates']
        self.samples_per_update: int = learner_config['samples_per_update']
        self.completed_updates: int = 0
        self.total_steps: int = 0
        self.batch_size: int = learner_config['batch_size']
        self.importance_sampling: bool = learner_config['IS']
        self.episodes_infos: List[Dict] = []
        self.total_episodes: int = 0
        self.sgd_iterations: int = learner_config['sgd_iterations']
        self.entropy_coeff: float = learner_config['entropy_coefficient']
        self.value_loss_coeff: float = learner_config['value_loss_coefficient']
        self.gamma: float = learner_config['gamma']
        self.gae_lambda: float = learner_config['lam']
        self.gae_horizon: int = learner_config['gae_horizon']
        self.clip_epsilon: float = learner_config['clip_epsilon']


    def _init_wandb(self, learner_config: Dict) -> None:
        """
        Initialize Weights & Biases for logging.
        """
        self.run_name = learner_config['run_name']
        wandb.init(project='AI4REALNET-T3.4', config=learner_config, mode='offline')
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='epoch')
        wandb.run.name = f"{self.run_name}_PPO"
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')
        wandb.watch(self.controller.encoder_network, log='all')


    def _init_normalisation(self) -> None:
        """ Normalisation setup """
        self.flatland_normalisation: FlatlandNormalisation = FlatlandNormalisation(
            n_nodes=self.controller.config['n_nodes'],
            n_features=self.controller.config['n_features'],
            n_agents=self.env.number_of_agents,
            env_size=(self.env_config.width, self.env_config.height)
        )


    def run(self) -> None:
        """
        Simple PPO training run.
        """
        # initialise learning rollout
        _ = self.env.reset()
        self.n_agents = self.env_config.get_num_agents()
        self.rollout = MultiAgentRolloutBuffer(n_agents=self.n_agents)
        self.rollout.reset(n_agents=self.n_agents)
        interrupted = False

        try:
            # TODO: gather rollouts and update when enough data is collected
            while self.completed_updates < self.target_updates:
                # gather rollouts
                log_info = self.gather_rollouts()
                wandb.log(log_info)
                if self.rollout.total_steps >= self.samples_per_update:
                    # update the controller with the current rollout
                    self._optimise()
                    self.completed_updates += 1
                    print(f'\n\nCompleted Updates: {self.completed_updates} / {self.target_updates}\n\n')
                    wandb.log({'train/average_episode_reward': np.mean([ep['average_episode_reward'] for ep in self.rollout.episodes])})


                    # reset rollout for next update
                    self.rollout.reset(n_agents=self.n_agents)

        except Exception as e:
            interrupted = True
            print(f'\nError received: {e}. Saving current model parameters before shutting down.\n')
        finally:
            wandb.finish()
            if interrupted:
                self._save_model()
            self._save_model()

    
    def gather_rollouts(self) -> Dict[str, Any]:
        episode_step = 0
        active_list = [True for _ in range(self.n_agents)]
        actions_dict = {i: 0 for i in range(self.n_agents)}

        current_state_dict, _ = self.env.reset()
        current_state_tensor = obs_dict_to_tensor(observation=current_state_dict, 
                                                  obs_type=self.obs_type, 
                                                  n_agents=self.n_agents,
                                                  max_depth=self.max_depth,
                                                  n_nodes=self.controller.config['n_nodes'])
        current_state_tensor = self.flatland_normalisation.normalise(current_state_tensor.unsqueeze(0)).squeeze(0)

        while any(active_list) and episode_step < self.max_steps_per_episode:
            actions, log_probs, state_values, _ = self.controller.sample_action(current_state_tensor)

            # reduce to active agents
            actions = actions[active_list]
            log_probs = log_probs[active_list]
            state_values = state_values[active_list]

            # create actions dict and step environment
            actions_dict = {idx: int(actions[i]) for i, idx in enumerate(actions)}
            next_state_dict, rewards, dones, infos = self.env.step(actions_dict)
            next_state_tensor = obs_dict_to_tensor(observation=next_state_dict, 
                                                   obs_type=self.obs_type, 
                                                   n_agents=self.n_agents,
                                                   max_depth=self.max_depth,
                                                   n_nodes=self.controller.config['n_nodes'])
            
            # reduce to active agents and normalise
            rewards = [rewards[i] for i in active_list if i]
            next_state_tensor = next_state_tensor[active_list]
            next_state_tensor = self.flatland_normalisation.normalise(next_state_tensor.unsqueeze(0)).squeeze(0).detach()
            next_state_values = self.controller.state_values(next_state_tensor, extras={}).detach()

            # Store transition in rollout buffer
            self.rollout.add_transitions(states=current_state_tensor.detach(),
                                         state_values=state_values.squeeze(1).detach(),
                                         next_states=next_state_tensor,
                                         next_state_values=next_state_values,
                                         actions=actions,
                                         log_probs=log_probs,
                                         rewards=rewards,
                                         dones=dones,
                                         extras={})

            current_state_tensor = next_state_tensor
            episode_step += 1
            self.total_steps += 1

            active_list = [not done for done in dones.values()][:-1]

        self.rollout.end_episode()
        log_info = {'episode': self.total_episodes,
                    'episode/total_reward': self.rollout.episodes[-1]['total_reward'],
                    'episode/average_reward': self.rollout.episodes[-1]['average_episode_reward'],
                    'episode/average_length': self.rollout.episodes[-1]['average_episode_length'],
                    'episode/completion': sum([dones[agent] for agent in range(self.n_agents)]) / self.n_agents}
        return log_info


    def _save_model(self, suffix: Optional[str] = None) -> None:
        """Persist the current controller parameters to disk."""
        savepath = os.path.join('models', f'{self.run_name}')
        if suffix:
            savepath = os.path.join(savepath, suffix)
        os.makedirs(savepath, exist_ok=True)
        torch.save(self.controller.actor_network.state_dict(), os.path.join(savepath, 'actor.pth'))
        torch.save(self.controller.critic_network.state_dict(), os.path.join(savepath, 'critic.pth'))
        torch.save(self.controller.encoder_network.state_dict(), os.path.join(savepath, 'encoder.pth'))
        print(f'Model parameters saved to {savepath}')


    def _build_optimiser(self, optimiser_config: Dict[str, Union[int, str]]) -> optim.Optimizer:
        if optimiser_config['type'] == 'adam': 
            self.optimiser = optim.Adam(params=chain(self.controller.actor_network.parameters(), self.controller.critic_network.parameters(), self.controller.encoder_network.parameters()), lr=float(optimiser_config['learning_rate']))
        else: 
            raise Warning('Only Adam optimiser has been implemented')


    def _optimise(self) -> Dict[str, List[float]]:
        """
        Optimise the model parameters using the collected rollouts.
        """
        losses = {
            'policy_loss': [],
            'value_loss': [],
            'total_loss': []
        }

        gae_stats = self._gaes()
        self.rollout.get_transitions(gae=True) # TODO: check that this is done correctly everywhere
        self._normalise_rewards()

        for sgd_iter in range(self.sgd_iterations):  
            entropy_sum: Tensor = Tensor()
            old_log_probs_sum: Tensor = Tensor()
            new_log_probs_sum: Tensor = Tensor()
            actor_delta = []
            critic_delta = []

            for minibatch in self.rollout.get_minibatches(self.batch_size): # TODO: check that this is the right batchsize
                new_log_probs, entropy, new_state_values, new_next_state_values = self._evaluate(minibatch['states'], minibatch['next_states'], minibatch['actions'])
                minibatch['new_log_probs'] = new_log_probs
                minibatch['new_state_values'] = new_state_values.squeeze(-1)
                minibatch['bootstrap_values'] = new_next_state_values.squeeze(-1)
                minibatch['entropy'] = entropy

                total_loss, actor_loss, critic_loss = self._loss(minibatch)
                
                #! delta actor / critic norm calculation
                with torch.no_grad():
                    actor_before = torch.nn.utils.parameters_to_vector(self.controller.actor_network.parameters()).norm().item()
                    critic_before = torch.nn.utils.parameters_to_vector(self.controller.critic_network.parameters()).norm().item()
                    encoder_before = torch.nn.utils.parameters_to_vector(self.controller.encoder_network.parameters()).norm().item()

                self.optimiser.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controller.get_parameters(), max_norm=5.0)
                self.optimiser.step()

                #! delta actor / critic norm calculation
                with torch.no_grad():
                    actor_after = torch.nn.utils.parameters_to_vector(self.controller.actor_network.parameters()).norm().item()
                    critic_after = torch.nn.utils.parameters_to_vector(self.controller.critic_network.parameters()).norm().item()
                    encoder_after = torch.nn.utils.parameters_to_vector(self.controller.encoder_network.parameters()).norm().item()
                    actor_delta = np.abs(actor_after - actor_before)
                    critic_delta = np.abs(critic_after - critic_before)
                    encoder_delta = np.abs(encoder_after - encoder_before)

                # add metrics
                losses['policy_loss'].append(actor_loss.item())
                losses['value_loss'].append(critic_loss.item())
                losses['total_loss'].append(total_loss.item())

                # entropy and log probs for approx KL
                entropy_sum = torch.cat((entropy_sum, minibatch['entropy'].detach()))
                old_log_probs_sum = torch.cat((old_log_probs_sum, minibatch['log_probs'].detach()))
                new_log_probs_sum = torch.cat((new_log_probs_sum, new_log_probs.detach()))

            self.epochs += 1
            approx_kl = torch.mean(old_log_probs_sum - new_log_probs_sum).item()
            wandb.log({
                'epoch': self.epochs,
                'train/policy_loss': np.mean(losses['policy_loss']),
                'train/value_loss': np.mean(losses['value_loss']),
                'train/total_loss': np.mean(losses['total_loss']),
                'train/raw_gae_mean': gae_stats[0],
                'train/raw_gae_std': gae_stats[1],
                'train/mean_entropy': entropy.mean().item(),
                'train/approx_kl': approx_kl,
                #! delta actor / critic norm calculation
                'train/actor_delta': actor_delta,
                'train/critic_delta': critic_delta,
                'train/encoder_delta': encoder_delta,
            })
            
            # Early stopping if KL divergence too high
            if approx_kl > 0.02:
                print(f"Early stopping at iteration {sgd_iter + 1}/{self.sgd_iterations} due to high KL: {approx_kl:.4f}")
                break

        return losses
    
    
    def _loss(self, minibatch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Calculate the loss for the actor and critic networks.
        """
        actor_loss = policy_loss(gae=minibatch['gaes'],
                                new_log_prob=minibatch['new_log_probs'],
                                old_log_prob=minibatch['log_probs'],
                                clip_eps=self.clip_epsilon)

        if self.importance_sampling:
            critic_loss = value_loss_with_IS(predicted_values=minibatch['new_state_values'],
                                             bootstrap_values=minibatch['bootstrap_values'],
                                            new_state_values=minibatch['new_state_values'],
                                            new_log_prob=minibatch['new_log_probs'],
                                            old_log_prob=minibatch['log_probs'],
                                            reward=minibatch['rewards'],
                                            done=minibatch['dones'],
                                            gamma=self.gamma
                                            )
        else:
            # Compute value targets from GAE + baseline
            value_targets = (minibatch['gaes'] + minibatch['state_values']).detach()
            critic_loss = value_loss(predicted_values=minibatch['new_state_values'],
                                    target_values=value_targets
                                    )

        # Entropy bonus
        entropy_loss = -minibatch['entropy'].mean()  # Encourage exploration

        # Total loss & optimisation step        # TODO: controller or learner config?
        total_loss: Tensor = actor_loss + critic_loss * self.value_loss_coeff + entropy_loss * self.entropy_coeff
        return total_loss, actor_loss, critic_loss


    def _gaes(self) -> Tuple[float, float]:
        # TODO: normalise GAEs
        for idx, episode in enumerate(self.rollout.episodes):
            self.rollout.episodes[idx]['gaes'] = [[] for _ in range(self.env_config.n_agents)]
            for agent in range(len(episode['states'])):
                state_values = torch.stack(episode['state_values'][agent]).detach()
                next_state_values = torch.stack(episode['next_state_values'][agent]).detach()

                rewards = torch.tensor(episode['rewards'][agent])
                dones = torch.tensor(episode['dones'][agent]).float()
                traj_len = len(rewards)

                gaes = [torch.tensor(0.0) for _ in range(len(rewards))]
                advantage = 0.0
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] + self.gamma * next_state_values[t] * (1 - dones[t]) - state_values[t]
                    advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * advantage
                    gaes[t] = advantage

                gae_tensor = torch.stack(gaes)
                self.rollout.episodes[idx]['gaes'][agent] = gae_tensor

        return self._normalise_gaes()

    
    def _normalise_gaes(self) -> Tuple[float, float]:
        n_episodes = len(self.rollout.episodes)

        stacked_gaes = torch.cat([torch.cat(self.rollout.episodes[episode]['gaes']) for episode in range(n_episodes)], dim=0)

        raw_gae_mean = stacked_gaes.mean()
        raw_gae_std = stacked_gaes.std().clamp_min(1e-8)  # avoid division by zero

        for idx in range(n_episodes):
            for agent in range(self.n_agents):
                gae_tensor = self.rollout.episodes[idx]['gaes'][agent]
                self.rollout.episodes[idx]['gaes'][agent] = (gae_tensor - raw_gae_mean) / raw_gae_std
        
        stacked_gaes = torch.cat([torch.cat(self.rollout.episodes[episode]['gaes']) for episode in range(n_episodes)], dim=0)
        gae_mean = stacked_gaes.mean().item()
        gae_std = stacked_gaes.std().item()

        return raw_gae_mean.item(), raw_gae_std.item(), gae_mean, gae_std
    

    def _normalise_rewards(self) -> None:
        """ Normalise rewards across all transitions in the rollout buffer. """
        rewards: Tensor = self.rollout.transitions['rewards']
        mean_reward = rewards.mean()   
        std_reward = rewards.std().clamp_min(1e-8)  # avoid division by zero
        normalised_rewards = (rewards - mean_reward) / std_reward
        self.rollout.transitions['rewards'] = normalised_rewards


    def _evaluate(self, states: Tensor, next_states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """ 
        Forward pass through the actor network to get the action distribution and log probabilities, to compute the entropy of the policy distribution and the current state values.
        """
        encoded_states = self.controller.encoder_network(states)
        encoded_next_states = self.controller.encoder_network(next_states)
        logits = self.controller.actor_network(encoded_states) # (batch_size, n_actions)
        action_distribution = torch.distributions.Categorical(logits=logits)
        log_probs = action_distribution.log_prob(actions) # (batch_size,)
        entropy = action_distribution.entropy() # (batch_size,)
        state_values = self.controller.critic_network(encoded_states) # (batch_size, 1)
        next_state_values = self.controller.critic_network(encoded_next_states) # (batch_size, 1)
        return log_probs, entropy, state_values, next_state_values
    
