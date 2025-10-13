# src/configs/EnvConfig.py
from typing import Dict, Union, Optional
from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen

from src.rewards.SimpleStepPenalty import SimpleStepPenalty

class FlatlandEnvConfig():
    def __init__(self, env_config: Dict[str, Union[int, float, dict, str]]):
        self.height: int = int(env_config['height'])
        self.width: int = int(env_config['width'])
        self.n_agents: int = int(env_config['n_agents'])
        self.n_cities: int = int(env_config['n_cities'])
        self.grid_distribution: bool = bool(env_config['grid_distribution'])
        self.max_rails_between_cities: int = int(env_config['max_rails_between_cities'])
        self.max_rail_pairs_in_city: int = int(env_config['max_rail_pairs_in_city'])
        self.observation_builder_config: Dict = env_config['observation_builder_config']
        self.malfunction_config: Optional[Dict[str, Union[float, int]]] = env_config.get('malfunction_config', None)
        self.speed_ratios: Dict[float, float] = env_config['speed_ratios']
        self.reward_config: Union[int, str, Dict] = env_config.get('reward_config', 0)
        self.random_seed: Optional[int] = env_config.get('random_seed', None)

        self.terminal_bonus: float = 0.0
        if isinstance(self.reward_config, dict) and self.reward_config.get('type', '').lower() == 'step_penalty':
            self.terminal_bonus = float(self.reward_config.get('terminal_bonus', 0.0))

    def _make_observation_builder(self):
        ob_cfg = self.observation_builder_config
        if ob_cfg['type'] == 'tree':
            predictor = None
            if ob_cfg.get('predictor', None) == 'shortest_path':
                predictor = ShortestPathPredictorForRailEnv()
            return TreeObsForRailEnv(max_depth=int(ob_cfg['max_depth']), predictor=predictor)
        elif ob_cfg['type'] == 'global':
            return GlobalObsForRailEnv()
        else:
            raise ValueError(f"Unknown observation builder type: {ob_cfg['type']}")

    def _make_reward_object(self):
        rc = self.reward_config
        if rc in (0, 'default', None):
            return None

        if isinstance(rc, dict):
            rtype = rc.get('type', '').lower()
            if rtype == 'step_penalty':
                step_penalty  = float(rc.get('step_penalty', -1.0))
                terminal_bonus = float(rc.get('terminal_bonus', 0.0))
                progress_coef = float(rc.get('progress_coef', 0.0))  # keep 0.0 to disable shaping
                self.terminal_bonus = terminal_bonus
                return SimpleStepPenalty(step_penalty=step_penalty,
                                         terminal_bonus=terminal_bonus,
                                         progress_coef=progress_coef)
            else:
                raise ValueError(f"Unknown reward_config type: {rtype}")

        if isinstance(rc, str) and rc.lower() == 'step_penalty':
            self.terminal_bonus = 0.0
            return SimpleStepPenalty(step_penalty=-1.0, terminal_bonus=0.0, progress_coef=0.0)

        raise ValueError(f"Unsupported reward_config: {rc}")

    def create_env(self) -> RailEnv:
        observation_builder = self._make_observation_builder()

        rail_generator = sparse_rail_generator(
            max_num_cities=self.n_cities,
            max_rails_between_cities=self.max_rails_between_cities,
            max_rail_pairs_in_city=self.max_rail_pairs_in_city,
            grid_mode=self.grid_distribution
        )
        line_generator = sparse_line_generator(self.speed_ratios)

        malfunction_generator = None
        if self.malfunction_config:
            mr = float(self.malfunction_config.get('malfunction_rate', 0.0))
            mn = int(self.malfunction_config.get('min_duration', 0))
            mx = int(self.malfunction_config.get('max_duration', 0))
            if mr > 0.0:
                malfunction_generator = ParamMalfunctionGen(
                    MalfunctionParameters(malfunction_rate=mr, min_duration=mn, max_duration=mx)
                )

        reward_obj = self._make_reward_object()

        common_kwargs = dict(
            width=self.width,
            height=self.height,
            number_of_agents=self.n_agents,
            rail_generator=rail_generator,
            line_generator=line_generator,
            malfunction_generator=malfunction_generator,
            obs_builder_object=observation_builder,
            random_seed=self.random_seed,
        )

        # Reward kwarg compatibility (newer: reward_builder, older: rewards)
        try:
            return RailEnv(**common_kwargs, reward_builder=reward_obj)
        except TypeError:
            return RailEnv(**common_kwargs, rewards=reward_obj)

    # UPDATE FUNCTIONS
    def update_random_seed(self, seed: int = 0) -> None:
        if seed:
            self.random_seed = int(seed)
        else:
            self.random_seed = 0 if self.random_seed is None else int(self.random_seed) + 1

    def update_observation_builder(self, observation_builder_config) -> None:
        self.observation_builder_config = observation_builder_config

    def update_malfunction_config(self, malfunction_config: Dict[str, Union[float, int]]) -> None:
        self.malfunction_config = malfunction_config

    def update_speed_ratios(self, speed_ratios: Dict[float, float]) -> None:
        self.speed_ratios = speed_ratios

    def update_reward_config(self, reward_config: Dict[str, Union[float, int, str]]) -> None:
        self.reward_config = reward_config
        if isinstance(self.reward_config, dict) and self.reward_config.get('type', '').lower() == 'step_penalty':
            self.terminal_bonus = float(self.reward_config.get('terminal_bonus', 0.0))
