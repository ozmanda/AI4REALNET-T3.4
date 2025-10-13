from typing import Dict, Tuple, Union
from flatland.envs.rail_env import RailEnv
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen
from flatland.envs.rewards import Rewards, DefaultRewards, BasicMultiObjectiveRewards, PunctualityRewards

from src.environments.scenario_loader import load_scenario_from_json, get_num_agents
from src.reward.reward_utils import SimpleReward
from src.reward.SimpleStepPenalty import SimpleStepPenalty

class FlatlandEnvConfig():
    def __init__(self, env_config: Dict[str, Union[int, float]]):
        if 'scenario_name' in env_config:
            self.scenario_name: str = env_config['scenario_name']
            self.observation_builder_config: Dict = env_config['observation_builder_config']
            self.scenario: bool = True
        else:
            self.scenario: bool = False
            self.height: int = env_config['height']
            self.width: int = env_config['width'] 
            self.n_agents: int = env_config['n_agents']
            self.n_cities: int = env_config['n_cities']
            self.grid_distribution: bool = env_config['grid_distribution']
            self.max_rails_between_cities: int = env_config['max_rails_between_cities']
            self.max_rail_pairs_in_city: int = env_config['max_rail_pairs_in_city']
            self.observation_builder_config: Dict = env_config['observation_builder_config']
            self.malfunction_config: Dict[str, Union[float, int]] = env_config['malfunction_config']
            self.speed_ratios: Dict[Dict[float, int], float] = env_config['speed_ratios']
            self.reward_config: int = env_config['reward_config']
            self.random_seed: int = env_config['random_seed'] if 'random_seed' in env_config else None

    def create_env(self): 
        """
        Returns a Flatland environment with the specified parameters.
        """
        self.create_observation_builder()
        if self.scenario: 
            return self.load_RailEnv()
        else:
            return self.create_RailEnv()


    def load_RailEnv(self) -> RailEnv:
        scenario_path = f"src/environments/{self.scenario_name}.json"
        railenv = load_scenario_from_json(scenario_path, self.observation_builder)
        self.n_agents = railenv.get_num_agents()
        self.height = railenv.height
        self.width = railenv.width
        return railenv
    

    def create_RailEnv(self) -> RailEnv:
        # Create the rail and line generator
        rail_generator = sparse_rail_generator(
            max_num_cities=self.n_cities,
            max_rails_between_cities=self.max_rails_between_cities,
            max_rail_pairs_in_city=self.max_rail_pairs_in_city,
            grid_mode=self.grid_distribution
        )
        line_generator = sparse_line_generator(self.speed_ratios)

        # Set malfunction generator
        if self.malfunction_config:
            malfunction_generator = ParamMalfunctionGen(MalfunctionParameters(malfunction_rate=self.malfunction_config['malfunction_rate'],
                                                                              min_duration=self.malfunction_config['min_duration'],
                                                                              max_duration=self.malfunction_config['max_duration']))
            
        rewards: Rewards = self.get_reward_function(self.reward_config)

        return RailEnv(width=self.width,
                       height=self.height,
                       number_of_agents=self.n_agents,
                       rail_generator=rail_generator,
                       line_generator=line_generator,
                       malfunction_generator=malfunction_generator,
                       obs_builder_object=self.observation_builder,
                       random_seed=self.random_seed,
                       rewards=rewards)

    def create_observation_builder(self) -> None:
        # Create the observation builder
        if self.observation_builder_config['type'] == 'tree':
            # Create the predictor
            if self.observation_builder_config['predictor'] == 'shortest_path':
                predictor = ShortestPathPredictorForRailEnv()
            self.observation_builder = TreeObsForRailEnv(max_depth=self.observation_builder_config['max_depth'],
                                                    predictor=predictor)
        elif self.observation_builder_config['type'] == 'global':
            self.observation_builder = GlobalObsForRailEnv()
        else:
            raise ValueError(f"Unknown observation builder type: {self.observation_builder_config['type']}")
        

    def get_reward_function(self, reward_config: str) -> Rewards:
        if reward_config: 
            if reward_config == 'simple':
                return SimpleReward()
            elif reward_config == 'simple_step_penalty':
                return SimpleStepPenalty(step_penalty=-1.0, terminal_bonus=10.0, progress_coef=0.1)
            elif reward_config == 'default':
                return DefaultRewards()
            elif reward_config == 'basic_multi_objective':
                return BasicMultiObjectiveRewards()
            elif reward_config == 'punctuality':
                return PunctualityRewards()
            else:
                raise ValueError(f"Unknown reward config: {reward_config}")
        else:
            return DefaultRewards()

    
    def get_num_agents(self) -> int:
        if self.scenario:
            return get_num_agents(f"src/environments/{self.scenario_name}.json")
        else:
            return self.n_agents    

    # UPDATE FUNCTIONS
    def update_random_seed(self, seed: int = 0) -> None:
        if seed:
            self.random_seed = seed
        else: 
            self.random_seed += 1

    def update_observation_builder(self, observation_builder_config) -> None:
        self.observation_builder_config = observation_builder_config

    def update_malfunction_config(self, malfunction_config: Dict[str, Union[float, int]]) -> None:
        self.malfunction_config = malfunction_config

    def update_speed_ratios(self, speed_ratios: Dict[float, float]) -> None:
        self.speed_ratios = speed_ratios

    def update_reward_config(self, reward_config: Dict[str, Union[float, int]]) -> None:
        self.reward_config = reward_config