# src/algorithms/PPO/PPOLearner.py
import os
import wandb
import queue
import numpy as np
from itertools import chain
from typing import List, Dict, Union, Tuple

import torch
from torch import Tensor
import torch.optim as optim
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.managers import DictProxy

from src.algorithms.PPO.PPOWorker import PPOWorker
from src.configs.ControllerConfigs import PPOControllerConfig
from src.controllers.PPOController import PPOController
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer
from src.algorithms.loss import value_loss, value_loss_with_IS, policy_loss
from src.configs.EnvConfig import FlatlandEnvConfig

class PPOLearner():
    def __init__(self, controller_config: PPOControllerConfig, learner_config: Dict, env_config: FlatlandEnvConfig, device: str = None) -> None:
        self.env_config = env_config
        self._init_learning_params(learner_config)
        self._init_controller(controller_config)

        self.n_workers: int = learner_config['n_workers']
        self.masking_config: Dict[str, Union[bool, str, float]] = learner_config.get('masking', {})
        # new: allow tuning non-decision weight α via YAML (default 0.2)
        self.non_decision_weight: float = float(self.masking_config.get('non_decision_weight', 0.2))
        self._init_queues()


        self.optimizer: optim.Optimizer = self._build_optimizer(learner_config['optimiser_config'])
        self.update_step: int = 0

        self._init_wandb(learner_config)

    def _init_controller(self, config: PPOControllerConfig) -> None:
        self.controller_config = config
        self.n_nodes: int = config.config_dict['n_nodes']
        self.state_size: int = config.config_dict['state_size']
        self.entropy_coeff: float = config.config_dict['entropy_coefficient']
        self.value_loss_coeff: float = config.config_dict['value_loss_coefficient']
        self.gamma: float = config.config_dict['gamma']
        self.controller = config.create_controller()

    def _init_learning_params(self, learner_config: Dict) -> None:
        self.max_steps: int = learner_config['max_steps']
        self.max_steps_per_episode: int = learner_config['max_steps_per_episode']
        self.target_updates: int = learner_config['target_updates']
        self.samples_per_update: int = learner_config['samples_per_update']
        self.completed_updates: int = 0
        self.total_steps: int = 0
        self.iterations: int = learner_config['training_iterations']
        self.batch_size: int = learner_config['batch_size']
        self.importance_sampling: bool = learner_config['IS']
        self.episodes_infos: List[Dict] = []
        self.total_episodes: int = 0

    def _init_wandb(self, learner_config: Dict) -> None:
        self.run_name = learner_config['run_name']
        wandb.init(project='AI4REALNET-T3.4', entity='CLS-FHNW', config=learner_config, reinit=True)
        wandb.run.define_metric('episodes/*', step_metric='episode')
        wandb.run.define_metric('train/*', step_metric='update_step')
        wandb.run.name = f"{self.run_name}_PPO"
        wandb.watch(self.controller.actor_network, log='all')
        wandb.watch(self.controller.critic_network, log='all')

    def _init_queues(self) -> None:
        self.logging_queue: mp.Queue = mp.Queue()
        self.rollout_queue: mp.Queue = mp.Queue()
        self.barrier = mp.Barrier(self.n_workers + 1)
        self.manager = mp.Manager()
        self.done_event: Event = mp.Event()

    def sync_run(self) -> None:
        self.shared_weights: DictProxy = self.manager.dict()
        self.rollout = MultiAgentRolloutBuffer(n_agents=self.env_config.n_agents)

        mp.set_start_method('spawn', force=True)
        workers: List[PPOWorker] = []
        print('Initialising workers...')
        for worker_id in range(self.n_workers):
            worker = PPOWorker(worker_id=worker_id,
                               logging_queue=self.logging_queue,
                               rollout_queue=self.rollout_queue,
                               shared_weights=self.shared_weights,
                               barrier=self.barrier,
                               done_event=self.done_event,
                               env_config=self.env_config,
                               controller_config=self.controller_config,
                               max_steps=(self.max_steps, self.max_steps_per_episode),
                               device='cpu',
                               masking_config=self.masking_config)
            workers.append(worker)
            worker.start()
        self._broadcast_controller_state()

        while self.completed_updates < self.target_updates:
            self.barrier.wait()
            for _ in range(self.n_workers):
                try:
                    self._gather_rollout()
                except Exception as e:
                    print(f"Error in _gather_rollout: {e}")
                    continue

            if self.rollout.total_steps >= self.samples_per_update:
                self._optimise()
                self.completed_updates += 1
                print(f'\n\nCompleted Updates: {self.completed_updates} / {self.target_updates}\n\n')

                avg_ret  = np.mean([ep.get('average_episode_return', ep.get('average_episode_reward', 0.0)) for ep in self.rollout.episodes])
                avg_step = np.mean([ep.get('average_step_reward', -999.0) for ep in self.rollout.episodes])
                succ_rate = float(np.mean([ep.get('success', 0) for ep in self.rollout.episodes])) if len(self.rollout.episodes) > 0 else 0.0
                wandb.log({
                    'train/average_episode_return': avg_ret,
                    'train/average_step_reward':   avg_step,
                    'train/average_episode_reward': avg_ret,
                    'train/success_rate':          succ_rate,
                })

                self._broadcast_controller_state()
                self.rollout.reset(n_agents=self.env_config.n_agents)

        self.done_event.set()
        for _ in range(self.n_workers):
            try:
                self._gather_rollout()
            except queue.Empty:
                continue
            except Exception:
                pass

        if getattr(self.rollout, "total_steps", 0) > 0:
            self._optimise()
            avg_ret  = np.mean([ep.get('average_episode_return', ep.get('average_episode_reward', 0.0)) for ep in self.rollout.episodes])
            avg_step = np.mean([ep.get('average_step_reward', -999.0) for ep in self.rollout.episodes])
            succ_rate = float(np.mean([ep.get('success', 0) for ep in self.rollout.episodes])) if len(self.rollout.episodes) > 0 else 0.0
            wandb.log({
                'train/average_episode_return': avg_ret,
                'train/average_step_reward':   avg_step,
                'train/average_episode_reward': avg_ret,
                'train/success_rate':          succ_rate,
            })

        for w in workers:
            w.join(timeout=5)
            if w.is_alive():
                w.terminate()

        wandb.finish()

        savepath = os.path.join('models', f'{self.run_name}')
        os.makedirs(savepath, exist_ok=True)
        torch.save(self.controller.actor_network.state_dict(), os.path.join(savepath, 'actor_model.pth'))
        torch.save(self.controller.critic_network.state_dict(), os.path.join(savepath, 'critic_model.pth'))

    def _gather_rollout(self) -> None:
        worker_rollout: Dict[str, List] = self.rollout_queue.get(timeout=120)
        self.rollout.add_episode(worker_rollout)
        self._log_episode_info()

    def _broadcast_controller_state(self) -> None:
        controller_state = (self.controller.actor_network.state_dict(),
                            self.controller.critic_network.state_dict())
        self.shared_weights['controller_state'] = controller_state
        self.shared_weights['update_step'] = self.update_step
        self.barrier.wait()

    def _log_episode_info(self):
        try:
            while True:
                log_info = self.logging_queue.get_nowait()
                worker_id = log_info['worker_id']
                wandb.log({
                    'episode': log_info.get('episode', 0),
                    f'worker_{worker_id}/episode_reward': log_info['episode/reward'],
                    f'worker_{worker_id}/episode_length': log_info['episode/average_length'],
                    f'worker_{worker_id}/episode_success': log_info.get('episode/success', 0),
                    'episodes/average_reward': log_info['episode/reward'],
                    'episodes/success': log_info.get('episode/success', 0),
                })
        except queue.Empty:
            pass

    def _build_optimizer(self, optimiser_config: Dict[str, Union[int, str]]) -> optim.Optimizer:
        if str(optimiser_config['type']).lower() == 'adam':
            return optim.Adam(
                params=chain(self.controller.actor_network.parameters(),
                             self.controller.critic_network.parameters()),
                lr=float(optimiser_config['learning_rate'])
            )
        raise ValueError('Only Adam optimiser has been implemented')

    def _optimise(self) -> Dict[str, List[float]]:
        losses = {'policy_loss': [], 'value_loss': []}

        self._gaes()
        self.rollout.get_transitions(gae=True)

        for epoch in range(self.iterations):
            for minibatch in self.rollout.get_minibatches(self.batch_size):
                if 'gaes' not in minibatch or minibatch['gaes'].numel() == 0:
                    wandb.log({'update_step': self.update_step,
                               'train/policy_loss': 0.0,
                               'train/value_loss': 0.0})
                    continue

                self.current_minibatch = {'action_mask': minibatch.get('action_mask', None)}

                new_log_probs, entropy, new_state_values, new_next_state_values = self._evaluate(
                    minibatch['states'], minibatch['next_states'], minibatch['actions'].long()
                )
                minibatch['new_log_probs'] = new_log_probs
                minibatch['new_state_values'] = new_state_values.squeeze(-1)
                minibatch['new_next_state_values'] = new_next_state_values.squeeze(-1)
                minibatch['entropy'] = entropy

                # diagnostics
                if 'decision_mask' in minibatch:
                    dm_for_log = minibatch['decision_mask'].float()
                    wandb.log({'update_step': self.update_step,
                               'train/decision_step_frac': float(dm_for_log.mean().item()),
                               'train/decision_steps': float(dm_for_log.sum().item())})
                if 'action_mask' in minibatch:
                    am = minibatch['action_mask'].to(torch.bool)
                    a = minibatch['actions'].long()
                    picked_valid = am.gather(1, a.view(-1,1)).squeeze(1)
                    invalid_frac = (~picked_valid).float().mean().item()
                    wandb.log({'update_step': self.update_step,
                               'train/invalid_action_frac': float(invalid_frac)})

                    # -------- advantage normalization with down-weighting of non-decision steps --------
                    adv = minibatch['gaes']
                    dm = minibatch.get('decision_mask', None)
                    alpha = getattr(self, 'non_decision_weight', 0.2)  # from YAML

                    if dm is not None:
                        dm = dm.to(adv.device).float()
                        sel = dm == 1
                        if sel.any():
                            mu = adv[sel].mean(); std = adv[sel].std(unbiased=False)
                        else:
                            mu = adv.mean();      std = adv.std(unbiased=False)
                        adv = (adv - mu) / (std + 1e-8)
                        weights = alpha + (1 - alpha) * dm   # α for non-decisions, 1 for decisions
                        adv = adv * weights
                        minibatch['entropy'] = minibatch['entropy'] * weights
                    else:
                        mu = adv.mean(); std = adv.std(unbiased=False)
                        adv = (adv - mu) / (std + 1e-8)
                    minibatch['gaes'] = adv
                    # ------------------------------------------------------------------------------------


                total_loss, actor_loss, critic_loss = self._loss(minibatch)
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                self.update_step += 1

                with torch.no_grad():
                    kl = minibatch['log_probs'] - minibatch['new_log_probs']
                    approx_kl = float(torch.nan_to_num(kl, nan=0.0).mean().abs().item())
                    ent_mean = float(torch.nan_to_num(minibatch['entropy'], nan=0.0).mean().item())

                wandb.log({
                    'update_step': self.update_step,
                    'train/policy_loss': float(actor_loss.item()),
                    'train/value_loss': float(critic_loss.item()),
                    'train/approx_kl': approx_kl,
                    'train/entropy_masked_mean': ent_mean,
                })

        self.current_minibatch = None
        return losses

    def _loss(self, minibatch: Dict[str, Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        actor_loss = policy_loss(gae=minibatch['gaes'],
                                 new_log_prob=minibatch['new_log_probs'],
                                 old_log_prob=minibatch['log_probs'],
                                 clip_eps=self.controller.config['clip_epsilon'])

        if self.importance_sampling:
            critic_loss = value_loss_with_IS(state_values=minibatch['new_state_values'],
                                             next_state_values=minibatch['new_next_state_values'],
                                             new_log_prob=minibatch['new_log_probs'],
                                             old_log_prob=minibatch['log_probs'],
                                             reward=minibatch['rewards'],
                                             done=minibatch['dones'],
                                             gamma=self.controller.config['gamma'])
        else:
            critic_loss = value_loss(state_values=minibatch['new_state_values'],
                                     next_state_values=minibatch['new_next_state_values'],
                                     reward=minibatch['rewards'],
                                     done=minibatch['dones'],
                                     gamma=self.controller.config['gamma'])

        entropy_loss = -minibatch['entropy'].mean()
        total_loss: Tensor = actor_loss + \
            critic_loss * self.controller.config['value_loss_coefficient'] + \
            entropy_loss * self.controller.config['entropy_coefficient']
        return total_loss, actor_loss, critic_loss

    def _gaes(self) -> None:
        with torch.no_grad():
            for idx, episode in enumerate(self.rollout.episodes):
                self.rollout.episodes[idx]['gaes'] = [[] for _ in range(self.env_config.n_agents)]
                for agent in range(len(episode['states'])):
                    state_values = torch.stack(episode['state_values'][agent])
                    next_state_values = torch.stack(episode['next_state_values'][agent])
                    rewards = torch.tensor(episode['rewards'][agent])
                    dones = torch.tensor(episode['dones'][agent]).float()
                    dones = (dones != 0).float()
                    gaes = []
                    deltas = rewards + self.gamma * next_state_values * (1 - dones) - state_values
                    gae = 0.0
                    for t in reversed(range(len(rewards))):
                        gae = deltas[t] + self.gamma * self.controller.config['lam'] * (1 - dones[t]) * gae
                        gaes.insert(0, gae)
                    self.rollout.episodes[idx]['gaes'][agent] = torch.stack(gaes)

    def _evaluate(self, states: Tensor, next_states: Tensor, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        action_mask = None
        if hasattr(self, 'current_minibatch') and self.current_minibatch is not None:
            action_mask = self.current_minibatch.get('action_mask', None)

        logits = self.controller.actor_network(states)

        if action_mask is not None:
            if action_mask.dtype != torch.bool:
                action_mask = action_mask.to(torch.bool)
            valid_any = action_mask.any(dim=1)
            if not torch.all(valid_any):
                fixed = action_mask.clone()
                fixed[~valid_any, 0] = True  # allow DO_NOTHING
                action_mask = fixed
            logits = logits.masked_fill(~action_mask.to(logits.device), -1e9)

        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions.long())
        entropy = dist.entropy()
        state_values = self.controller.critic_network(states)
        next_state_values = self.controller.critic_network(next_states)
        return log_probs, entropy, state_values, next_state_values
