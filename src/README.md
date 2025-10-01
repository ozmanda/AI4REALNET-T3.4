# Main Code

This codebase is structured as follows: 
- **algorithms:** algorithm implementations and configurations, each consisting of the following: 
    - **Learner:** contains all learning logic, updates the policy based on experiences gathered from rollouts.
    - **Worker:** a parallel instance which manages a ``Runner`` and returns experiences in the form of ``Rollouts``
- **configs:** contains all config classes and ``.yaml`` config files.
- **controllers:** interacts with the environment according to the current policy
- **environments:** contains file and scripts to initialise flatland environments for various testing scenarios
- **memory:** various memory classes, including the default ``MultiAgentRolloutBuffer``
- **negotiation:** contains all classes and functions relating to agent negotiation.
- **networks:** network modules used to construct controllers.
- **utils:** utility functions, including the graph representation of the environment and associated functions.