from flatland.envs.rail_env import RailEnv
from flatland.envs.observations import TreeObsForRailEnv, GlobalObsForRailEnv
from flatland.envs.predictions import ShortestPathPredictorForRailEnv
from flatland.envs.rail_generators import sparse_rail_generator
from flatland.envs.line_generators import sparse_line_generator
from flatland.envs.malfunction_generators import MalfunctionParameters, ParamMalfunctionGen


def small_flatland_env(observation: str = 'tree', max_tree_depth: int = 2, malfunctions: bool = True) -> RailEnv:
    """
    RailEnv parameters:
        width = 35
        height = 35
        cities = 4
        trains = 8
        city_grid_distribution = False
        max_rails_between_cities = 2
        max_rail_pairs_in_city = 2
    
    Observation Builder:
        predictor = ShortestPathPredictorForRailEnv()

    Malfunctions are optional and can be turned off.    
    """
    if observation == 'tree':
        observation_builder = TreeObsForRailEnv(max_depth=max_tree_depth, predictor=ShortestPathPredictorForRailEnv())
    elif observation == 'global':
        observation_builder = GlobalObsForRailEnv()

    rail_generator = sparse_rail_generator(
        max_num_cities=4,
        max_rails_between_cities=2,
        max_rail_pairs_in_city=2,
        grid_mode=False
    )

    speed_ratio_map = {1.: 0.7,
                       0.5: 0.3}
    
    line_generator = sparse_line_generator(speed_ratio_map)

    if malfunctions:
        stochastic_malfunctions = MalfunctionParameters(malfunction_rate=1/1000,
                                                        min_duration=20,
                                                        max_duration=50)
        malfunction_generator = ParamMalfunctionGen(stochastic_malfunctions)
    else:
        malfunction_generator = None
    
    return RailEnv(width=35,
                   height=28,
                   number_of_agents=8,
                   rail_generator=rail_generator,
                   line_generator=line_generator,
                   malfunction_generator=malfunction_generator,
                   obs_builder_object=observation_builder
                   )