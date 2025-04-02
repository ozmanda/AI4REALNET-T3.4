# Algorithms

The following algorithms have been been implemented for experimentation in T3.4: 
1. PPO 
2. IC3Net 

# Structure
Each algorithm has the following elements: 
- **Configs:** class wrapping the configuration of the algorithm.
- **Controller:** class that executes the current policy and interacts with the environment.
- **Learner:** executes the learning logic, is called by the main training process.
- **Rollout:** a class to gather the experience tuples for learning. 
- **Runner:** a class that coordinates rollout execution and passes data to the learner.
- **Worker:** a class allowing for the parallel execution of runners.