# Main Code

This codebase is built modularly: 
- **algorithms:** algorithm implementations and configurations, each consisting of the following: 
    - **Controller:** interacts with the environment according to the current policy.
    - **Learner:** contains all learning logic, updates the policy based on experiences gathered from rollouts.
    - **Worker:** a parallel instance which manages a ``Runner`` and returns experiences in the form of ``Rollouts``
    - **Runner:** uses the ``Controller`` to gather ``Rollouts``.
    - **Rollout:** gathers experience tuples from interaction with the environment.
- **configs:** contains all config classes and .yaml config files.
- **evaluation:** scripts and functions to support evaluation
- **networks:** network modules accessed by the algorithms
- **training:** main training scripts 
- **utils:** utility functions