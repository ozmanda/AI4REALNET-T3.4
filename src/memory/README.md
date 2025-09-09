# MultiAgentRolloutBuffer
The ``MultiAgentRolloutBuffer`` gathers the trajectories / transitions from multi-agent simulation environments over multiple episodes and contains functions to batch these trajectories for learning. 

## General Structure
Each ``MultiAgentRolloutBuffer`` is initiated with the number of agents in the environment and can be reset with a new number of agents using the ``.reset()`` function. During runtime, agent transitions are saved into ``.current_episode`` using the ``.add_transition`` function. Upon completion, the current episode can be reset by ``.reset_current_episode()``. Each ``episode`` contains the following values:

```python
self.current_episode: Dict[str, List] = {
    states: List[Tensor],
    state_values: List[Tensor],
    actions: List[Tensor],
    log_probs: List[Tensor],
    rewards: List[Tensor],
    next_states: List[Tensor],
    next_state_values: List[Tensor],
    dones: List[int],
    episode_length: List[int],
    episode_reward: List[float]
    average_episode_length: float,
    average_episode_reward: float
}
```

Upon episode completeion, the ``end_episode()`` function calculates the agent and average episode length and rewards, then adds it to the list of all completed episodes ``episodes: List[Dict]``. 