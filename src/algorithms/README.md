# Algorithms

The following algorithms have been been implemented for experimentation in T3.4: 
1. **IMPALA** for asynchronous training - from [Espeholt et al. (2018)](https://arxiv.org/abs/1802.01561)
2. **PPO** for synchronous training - from [Schulman et al. (2017)](https://arxiv.org/abs/1707.06347)
3. JBR_HSE_PPO - from [jbr-ai-labs Github](https://github.com/jbr-ai-labs/NeurIPS2020-Flatland-Competition-Solution)
4. IC3Net - from [Singh et al. (2018)](https://arxiv.org/abs/1812.09755)

# Structure
Each algorithm has the following elements: 
- **Configs:** class wrapping the configuration of the algorithm, found in ``src.conifgs``, along with the ``.yaml`` files for run configuration.
- **Controller:** class that executes the current policy and interacts with the environment.
- **Learner:** executes the learning logic, is called by the main training process.
- **Worker:** a class allowing for the parallel execution of rollout collection.