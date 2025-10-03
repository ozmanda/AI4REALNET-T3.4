import argparse
from argparse import Namespace
import numpy as np
import random
from flatland.envs.rail_env import RailEnv
from archive.loader import load_policy
from yaml import load, safe_load
from src.utils.observation.obs_utils import normalise_observation
from flatland.envs.rail_env import RailEnvActions
from collections import deque
from environments.env_small import small_flatland_env
import matplotlib.pyplot as plt


def train_agent(n_epsiodes: int, env: RailEnv, policyname: str, training_params: dict, seed=42):
    # random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # exploration parameters
    eps_start = training_params.pop("eps_start")
    eps_end = training_params.pop("eps_end")
    eps_decay = training_params.pop("eps_decay")

    training_params = Namespace(**training_params)

    # reset environment
    env.reset()

    # Calculate state size using the observation features and tree depth
    n_features_per_node = env.obs_builder.observation_dim
    n_nodes = np.sum([np.power(4, i) for i in range(env.obs_builder.max_depth + 1)])
    state_size = n_features_per_node * n_nodes
    action_size = 5

    # set maximum steps per episode
    max_steps = int(4 * 2 * (env.height + env.width + len(env.agents)))
    env._max_episode_steps = max_steps

    # progress tracking and variables
    action_dict = dict()
    scores_window = deque(maxlen=100)
    completion_window = deque(maxlen=100)
    scores = []
    action_count = [0] * action_size
    agent_obs = [None] * env.get_num_agents()
    agent_prev_obs = [None] * env.get_num_agents()
    agent_prev_action = [2] * env.get_num_agents()
    update_values = [False] * env.get_num_agents()

    policy = load_policy(policyname, state_size, action_size, training_params)

    for episode_idx in range(n_epsiodes):
        score = 0

        # Reset environment
        obs, info = env.reset(regenerate_rail=True, regenerate_schedule=True)

        # enter env
        for agent in env.get_agent_handles():
            action_dict.update({agent: RailEnvActions.MOVE_FORWARD})
        _, _, _, info = env.step(action_dict)

        # build agent-specific observations
        for agent in env.get_agent_handles():
            if obs[agent]:
                agent_obs[agent] = normalise_observation(obs[agent], env.obs_builder.max_depth, observation_radius=10)
                agent_prev_obs[agent] = agent_obs[agent].copy()
        
        # Run episode
        for step in range(max_steps - 1): 
            for agent in env.get_agent_handles():
                if info['action_required'][agent]:
                    update_values[agent] = True
                    action = policy.act(agent_obs[agent], eps=eps_start)
                    action_count[action] += 1
                else: 
                    update_values[agent] = False
                    action = 0
                action_dict.update({agent: action})

            # Environment step
            next_obs, all_rewards, done, info = env.step(action_dict)

            # Update replay buffer and train agent
            for agent in range(env.get_num_agents()):
                # only update the values when we are done or an action was taken 
                if update_values[agent] or done[agent]:
                    policy.step(agent_prev_obs[agent], agent_prev_action[agent], all_rewards[agent], agent_obs[agent], done[agent])

                    agent_prev_obs[agent] = agent_obs[agent].copy()
                    agent_prev_action[agent] = action_dict[agent]

                    if next_obs[agent]:
                        agent_obs[agent] = normalise_observation(next_obs[agent], env.obs_builder.max_depth, observation_radius=10)
                    
                    score += all_rewards[agent]
            
            if done['__all__']:
                break
                
        # Epsilon decay
        eps_start = max(eps_end, eps_decay * eps_start)
        action_probs = action_count / np.sum(action_count)

        # Collection information about training
        tasks_finished = np.sum([int(done[idx]) for idx in env.get_agent_handles()])
        completion_window.append(tasks_finished / max(1, env.get_num_agents()))
        scores_window.append(score / (max_steps * env.get_num_agents()))
        scores.append(np.mean(scores_window))
        action_probs = action_count / np.sum(action_count)

        print('\rTraining {} agents on {}x{}\t Episode {}\t Average Score: {:.3f}\tDones: {:.2f}%\tEpsilon: {:.2f} \t Action Probabilities: \t {}'.format(
        env.get_num_agents(),
        env.width, env.height,
        episode_idx,
        np.mean(scores_window),
        100 * np.mean(completion_window),
        eps_start,
        action_probs
        ))

    # Plot overall training progress at the end
    plt.plot(scores)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_params", type=str, help="Path to training parameters", default="training_parameters.yaml")
    parser.add_argument("--policy", type=str, help='Policy to be used', default="DDQN")
    args = parser.parse_args()

    # load training parameters
    with open(args.training_params, "r") as file:
        training_params = safe_load(file)

    n_episodes = training_params.pop("n_episodes")
    seed = training_params.pop("seed")
    run_name = training_params.pop("run_name")

    env: RailEnv = small_flatland_env(malfunctions=True)
    train_agent(n_episodes, env, args.policy, training_params, seed=seed)
