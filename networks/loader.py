def load_policy(policyname: str, state_size: int, action_size: int, training_params: dict):
    if policyname == "DDQN":
        from networks.DDQN import DDDQNPolicy
        return DDDQNPolicy(state_size, action_size, training_params)
    else:
        raise ValueError("Invalid policy name")