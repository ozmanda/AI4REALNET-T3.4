from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from typing import Dict, Tuple, Union
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen

class FlatlandEnvConfig():
    def __init__(self, env_config: Dict[str, Union[int, float]]):
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
        # Create the observation builder
        if self.observation_builder_config['type'] == 'tree':
            # Create the predictor
            if self.observation_builder_config['predictor'] == 'shortest_path':
                predictor = ShortestPathPredictorForRailEnv()
            observation_builder = TreeObsForRailEnv(max_depth=self.observation_builder_config['max_depth'],
                                                    predictor=predictor)
        elif self.observation_builder_config['type'] == 'global':
            observation_builder = GlobalObsForRailEnv()
            
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
        return RailEnv(width=self.width,
                       height=self.height,
                       number_of_agents=self.n_agents,
                       rail_generator=rail_generator,
                       line_generator=line_generator,
                       malfunction_generator=malfunction_generator,
                       obs_builder_object=observation_builder,
                       random_seed=self.random_seed)
    

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