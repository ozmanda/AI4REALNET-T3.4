from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from typing import Dict, Tuple, Union
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen

class FlatlandEnvConfig():
    def __init__(self,
                 height: int = 30, 
                 width: int = 30,
                 n_agents: int = 8,
                 n_cities: int = 4, 
                 grid_distribution: bool = False,
                 max_rails_between_cities: int = 2,
                 max_rail_pairs_in_city: int = 2,
                 observation_builder_config: Dict = None,
                 malfunction_config: Dict[str, Union[float, int]] = None,
                 speed_ratios: Dict[float, float] = None,
                 reward_config: Dict = None, 
                 random_seed = 42):
        self.height = height
        self.width = width 
        self.n_agents = n_agents
        self.n_cities = n_cities
        self.grid_distribution = grid_distribution
        self.max_rails_between_cities = max_rails_between_cities
        self.max_rail_pairs_in_city = max_rail_pairs_in_city
        self.observation_builder_config = observation_builder_config
        self.malfunction_config = malfunction_config
        self.reward_config = reward_config
        self.random_seed = random_seed

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
        line_generator = sparse_line_generator(self.observation_builder_config['speed_ratios'])

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
    def update_random_seed(self, seed: int) -> None:
        self.random_seed = seed

    def update_observation_builder(self, observation_builder_config) -> None:
        self.observation_builder_config = observation_builder_config

    def update_malfunction_config(self, malfunction_config: Dict[str, Union[float, int]]) -> None:
        self.malfunction_config = malfunction_config

    def update_speed_ratios(self, speed_ratios: Dict[float, float]) -> None:
        self.speed_ratios = speed_ratios

    def update_reward_config(self, reward_config: Dict[str, Union[float, int]]) -> None:
        self.reward_config = reward_config