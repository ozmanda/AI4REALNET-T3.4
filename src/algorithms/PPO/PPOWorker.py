# src/algorithms/PPO/PPOWorker.py
from collections import namedtuple
from typing import Tuple, Dict, Union, List, Optional
from flatland.envs.rail_env import RailEnv

import time
import numpy as np
import torch
from torch import Tensor
import torch.multiprocessing as mp
from multiprocessing.managers import DictProxy
from multiprocessing.synchronize import Event

from src.utils.obs_utils import obs_dict_to_tensor
from src.configs.EnvConfig import FlatlandEnvConfig
from src.controllers.PPOController import PPOController
from src.configs.ControllerConfigs import PPOControllerConfig
from src.memory.MultiAgentRolloutBuffer import MultiAgentRolloutBuffer

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward', 'next_state', 'done', 'info'))
# 0=DO_NOTHING, 1=LEFT, 2=FORWARD, 3=RIGHT, 4=STOP_MOVING
_DIR2DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

def _forward_is_occupied(env: RailEnv, row: int, col: int, direction: int, me_handle: int) -> bool:
    """True if the next forward cell is currently occupied by another agent."""
    if row is None or col is None or direction is None:
        return False
    dr, dc = _DIR2DELTA[int(direction)]
    r2, c2 = row + dr, col + dc
    occ = {(ag.position[0], ag.position[1]) for ag in env.agents if ag.handle != me_handle and ag.position is not None}
    return (r2, c2) in occ


def _get_from_agent_dict(d: Dict, idx: int, default=None):
    if idx in d:
        return d[idx]
    s = str(idx)
    if s in d:
        return d[s]
    s2 = f"agent_{idx}"
    if s2 in d:
        return d[s2]
    return default

def _set_in_agent_dict(d: Dict, idx: int, value):
    if idx in d:
        d[idx] = value
    elif str(idx) in d:
        d[str(idx)] = value
    elif f"agent_{idx}" in d:
        d[f"agent_{idx}"] = value
    else:
        d[idx] = value

class PPOWorker(mp.Process):
    def __init__(self,
                 worker_id: Union[str, int],
                 env_config: FlatlandEnvConfig,
                 controller_config: PPOControllerConfig,
                 logging_queue: mp.Queue,
                 rollout_queue: mp.Queue,
                 shared_weights,
                 barrier,
                 done_event: Event,
                 max_steps: Tuple = (10000, 1000),
                 device: str = 'cpu',
                 masking_config: Optional[Dict[str, Union[bool, str]]] = None):
        super().__init__()
        self.worker_id = worker_id
        self.logging_queue = logging_queue
        self.rollout_queue = rollout_queue
        self.shared_weights: DictProxy = shared_weights
        self.barrier = barrier
        self.done_event: Event = done_event
        self.device = device
        self.local_update_step: int = -1

        self.env_config: FlatlandEnvConfig = env_config
        self._init_env()
        self.controller_config: PPOControllerConfig = controller_config
        self.controller: PPOController = self.controller_config.create_controller()

        self.max_steps: int = max_steps[0]
        self.max_steps_per_episode: int = max_steps[1]
        self.rollout: MultiAgentRolloutBuffer = MultiAgentRolloutBuffer(n_agents=self.env.number_of_agents)

        # terminal bonus
        tb = getattr(self.env_config, "terminal_bonus", None)
        if tb is None:
            rb = getattr(self.env, "reward_builder", None)
            tb = float(getattr(rb, "terminal_bonus", 0.0)) if rb is not None else 0.0
        self.terminal_bonus: float = float(tb)
        self._done_once: List[bool] = [False] * self.env.number_of_agents

        # masking (relaxed)
        mc = masking_config or {}
        self.use_action_mask = bool(mc.get('use_action_mask', True))
        self.mask_wait_stop_if_movable = bool(mc.get('mask_wait_stop_if_movable', False))
        self.use_decision_mask = bool(mc.get('use_decision_mask', False))
        self.decision_rule = str(mc.get('decision_rule', 'turn_or_blocked')).lower()
        self.enforce_do_nothing_if_no_decision = bool(mc.get('enforce_do_nothing_if_no_decision', False))

    def _init_env(self) -> RailEnv:
        self.obs_type: str = self.env_config.observation_builder_config['type']
        self.max_depth: int = self.env_config.observation_builder_config['max_depth']
        self.env: RailEnv = self.env_config.create_env()

    def _safe_env_reset(self, max_tries: int = 10):
        last_err = None
        for attempt in range(max_tries):
            try:
                return self.env.reset()
            except ValueError as e:
                last_err = e
                new_seed = int(np.random.randint(0, 1_000_000)) + int(self.worker_id) * 1000 + attempt
                try:
                    self.env_config.update_random_seed(new_seed)
                    self.env = self.env_config.create_env()
                except Exception:
                    time.sleep(0.05)
        raise last_err

    # masks
    def _raw_valid_mask_single(self, pos, direction) -> List[bool]:
        mask = [False, False, False, False, False]
        if pos is None or direction is None:
            mask[0] = True; mask[2] = True; mask[4] = True
            return mask
        trans = self.env.rail.get_transitions(pos[0], pos[1], direction)
        left_dir, fwd_dir, right_dir = (direction + 3) % 4, direction, (direction + 1) % 4
        mask[1] = bool(trans[left_dir]); mask[2] = bool(trans[fwd_dir]); mask[3] = bool(trans[right_dir])
        mask[0] = True; mask[4] = True
        return mask

    def _apply_wait_stop_rule(self, mask: List[bool]) -> List[bool]:
        if not self.mask_wait_stop_if_movable:
            return mask
        any_move = mask[1] or mask[2] or mask[3]
        if any_move:
            mask[0] = False; mask[4] = False
        return mask

    def _valid_action_mask_single(self, pos, direction) -> List[bool]:
        return self._apply_wait_stop_rule(self._raw_valid_mask_single(pos, direction))

    def _valid_action_mask(self) -> torch.BoolTensor:
        masks = []
        for ag in self.env.agents:
            masks.append(self._valid_action_mask_single(ag.position, ag.direction))
        return torch.tensor(masks, dtype=torch.bool)

    # --- replace the original function with this one ---
    def _decision_mask_from_valid(self, valid_masks: torch.BoolTensor) -> torch.FloatTensor:
        """
        Decision if:
        - a turn is possible (left OR right), OR
        - forward is geometrically invalid, OR
        - we are at a switch (>= 2 valid geometric moves), OR
        - forward is valid but the next forward cell is OCCUPIED by another agent
        """
        if not self.use_decision_mask or self.decision_rule == 'none':
            return torch.ones(valid_masks.shape[0], dtype=torch.float32)

        left_ok, fwd_ok, right_ok = valid_masks[:, 1], valid_masks[:, 2], valid_masks[:, 3]

        # switch = at least two of {left, forward, right} are allowed by track geometry
        num_moves = left_ok.to(torch.int32) + fwd_ok.to(torch.int32) + right_ok.to(torch.int32)
        is_switch = num_moves >= 2

        # forward-occupied test per agent
        fwd_occ_list = []
        for ag in self.env.agents:
            if ag.position is None or ag.direction is None:
                fwd_occ_list.append(False)
            else:
                fwd_occ_list.append(_forward_is_occupied(self.env, ag.position[0], ag.position[1], ag.direction, ag.handle))
        fwd_occupied = torch.tensor(fwd_occ_list, dtype=torch.bool)

        decision = (left_ok | right_ok) | (~fwd_ok) | is_switch | (fwd_ok & fwd_occupied)
        return decision.float()


    def run(self) -> MultiAgentRolloutBuffer:
        current_state_dict, _ = self._safe_env_reset()
        self.rollout.reset(n_agents=self.env.number_of_agents)
        self._done_once = [False] * self.env.number_of_agents
        self._wait_for_weights()

        episode_step = 0
        self.total_episodes = 0
        n_agents = self.env.number_of_agents

        current_state_tensor: Tensor = obs_dict_to_tensor(
            observation=current_state_dict,
            obs_type=self.obs_type,
            n_agents=n_agents,
            max_depth=self.max_depth,
            n_nodes=self.controller.config['n_nodes']
        )

        while not self.done_event.is_set():
            action_mask = self._valid_action_mask()
            decision_mask = self._decision_mask_from_valid(action_mask)

            with torch.no_grad():
                logits = self.controller._make_logits(current_state_tensor)
                masked_logits = logits.masked_fill(~action_mask.to(logits.device), -1e9) if self.use_action_mask else logits
                if self.use_action_mask:
                    valid_any = action_mask.any(dim=1)
                    if not torch.all(valid_any):
                        fix = action_mask.clone()
                        fix[~valid_any, 0] = True
                        action_mask = fix
                        masked_logits = logits.masked_fill(~action_mask.to(logits.device), -1e9)
                dist = torch.distributions.Categorical(logits=masked_logits)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
                values = self.controller.critic_network(current_state_tensor)

            if self.enforce_do_nothing_if_no_decision and self.use_decision_mask:
                force_idx = (decision_mask == 0).nonzero(as_tuple=False).view(-1)
                if force_idx.numel() > 0:
                    actions[force_idx] = 0
                    dist2 = torch.distributions.Categorical(logits=masked_logits)
                    lp0 = dist2.log_prob(torch.zeros_like(actions))
                    log_probs[force_idx] = lp0[force_idx]

            actions_dict: Dict[Union[int, str], Tensor] = {i: actions[i].detach() for i in range(n_agents)}
            next_state, rewards, dones, infos = self.env.step(actions_dict)

            # terminal bonus once per agent
            if self.terminal_bonus != 0.0:
                for i in range(n_agents):
                    done_i = bool(_get_from_agent_dict(dones, i, False))
                    if done_i and not self._done_once[i]:
                        r_i = float(_get_from_agent_dict(rewards, i, 0.0))
                        _set_in_agent_dict(rewards, i, r_i + self.terminal_bonus)
                        self._done_once[i] = True

            next_state_tensor: Tensor = obs_dict_to_tensor(
                observation=next_state,
                obs_type=self.obs_type,
                n_agents=n_agents,
                max_depth=self.max_depth,
                n_nodes=self.controller.config['n_nodes']
            )
            with torch.no_grad():
                next_state_values = self.controller.state_values(next_state_tensor, extras=None)

            extras_out = {'action_mask': action_mask.to(torch.bool)}
            if self.use_decision_mask:
                extras_out['decision_mask'] = decision_mask

            self.rollout.add_transitions(
                states=current_state_tensor.detach(),
                actions=actions_dict,
                log_probs=log_probs.detach(),
                rewards=rewards,
                next_states=next_state_tensor.detach(),
                state_values=values.squeeze(-1).detach(),
                next_state_values=next_state_values.squeeze(-1).detach(),
                dones=dones,
                extras=extras_out
            )

            current_state_tensor = next_state_tensor
            episode_step += 1

            hard_done = all(bool(_get_from_agent_dict(dones, i, False)) for i in range(n_agents))
            time_limit = episode_step >= self.max_steps_per_episode
            if hard_done or time_limit:
                self.rollout.current_episode['success'] = int(all(self._done_once))
                self.rollout.end_episode()

                current_state_dict, _ = self._safe_env_reset()
                current_state_tensor = obs_dict_to_tensor(
                    observation=current_state_dict,
                    obs_type=self.obs_type,
                    n_agents=n_agents,
                    max_depth=self.max_depth,
                    n_nodes=self.controller.config['n_nodes']
                )
                episode_step = 0
                self.total_episodes += 1
                self._done_once = [False] * n_agents

                self.rollout_queue.put(self.rollout.episodes[-1])
                self.logging_queue.put({
                    'worker_id': self.worker_id,
                    'episode': self.total_episodes,
                    'episode/reward': self.rollout.episodes[-1]['average_episode_reward'],
                    'episode/average_length': self.rollout.episodes[-1]['average_episode_length'],
                    'episode/success': self.rollout.episodes[-1].get('success', 0),
                })
                if not self.done_event.is_set():
                    self._wait_for_weights()

    def _wait_for_weights(self):
        self.barrier.wait()
        if self.shared_weights['update_step'] > self.local_update_step:
            self.local_update_step = self.shared_weights['update_step']
            self.controller.update_weights(self.shared_weights['controller_state'])
