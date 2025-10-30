import os
import torch
import pytest
import numpy as np
import imageio.v2 as imageio
from datetime import datetime
from typing import List, Optional
from flatland.utils.rendertools import RenderTool
import unittest


from src.configs.ControllerConfigs import PPOControllerConfig
from src.configs.EnvConfig import FlatlandEnvConfig
from src.utils.file_utils import load_config_file
from src.utils.observation.obs_utils import calculate_state_size, obs_dict_to_tensor
from src.utils.observation.normalisation import FlatlandNormalisation

class ModelBehaviourTest(unittest.TestCase):
    def _load_controller(self, config: dict, env_config: FlatlandEnvConfig, device: torch.device) -> torch.nn.Module:
        controller_cfg = config['controller_config'].copy()
        n_nodes, state_size = calculate_state_size(env_config.observation_builder_config['max_depth'])
        controller_cfg['n_nodes'] = n_nodes
        controller_cfg['state_size'] = state_size

        controller = PPOControllerConfig(controller_cfg).create_controller()
        controller.to(device)
        controller.eval()

        
        encoder_path = os.path.join(self.modelpath, 'encoder.pth')
        actor_path = os.path.join(self.modelpath, 'actor.pth')
        critic_path = os.path.join(self.modelpath, 'critic.pth')

        if not (os.path.exists(encoder_path) and os.path.exists(actor_path) and os.path.exists(critic_path)):
            self.skipTest(
                f'Model checkpoints not found in {self.modelpath}. '
                'Train the learner or set MODEL_BEHAVIOUR_MODEL_DIR to an existing checkpoint directory.'
            )

        controller.encoder_network.load_state_dict(torch.load(encoder_path, map_location=device, weights_only=True))
        controller.actor_network.load_state_dict(torch.load(actor_path, map_location=device, weights_only=True))
        controller.critic_network.load_state_dict(torch.load(critic_path, map_location=device, weights_only=True))
        return controller


    def _render_frame(self, renderer: Optional[RenderTool], show: bool, frames: List[np.ndarray]) -> None:
        if renderer is None:
            return
        renderer.render_env(show=show, show_observations=False, show_predictions=False)
        frame = renderer.get_image()
        if frame is not None:
            frames.append(np.array(frame))


    def _save_video(self, frames: List[np.ndarray], output_path: str, fps: int) -> None:
        if not frames:
            return
        if imageio is None:
            self.skipTest('imageio is required to export behaviour videos.')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        imageio.mimsave(output_path, frames, fps=fps)
        print(f'Behaviour video saved to {output_path}')


    def test_model_behaviour(self) -> None:
        """
        Roll out a single episode with the trained controller, rendering each step so the agent behaviour can be inspected.
        Set MODEL_BEHAVIOUR_SAVE_VIDEO=1 to export a video to MODEL_BEHAVIOUR_VIDEO_PATH (defaults to test/renders/).
        """
        self.modelpath = 'models/FNN_large_env_run1'  # Adjust this path as needed
        config_path = 'src/configs/PPO_FNN.yaml'
        device = torch.device('cpu')
        show_render = 1 
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        video_path = f'{self.modelpath}/render_{timestamp}.mp4'
        render_fps = 6

        config = load_config_file(config_path)
        env_config = FlatlandEnvConfig(config['environment_config'])
        env = env_config.create_env()
        controller = self._load_controller(config, env_config, device)

        normaliser = FlatlandNormalisation(
            n_nodes=controller.config['n_nodes'],
            n_features=controller.config['n_features'],
            n_agents=env.number_of_agents,
            env_size=(env_config.width, env_config.height),
        )

        renderer: Optional[RenderTool] = None
        if show_render:
            renderer = RenderTool(
                env,
                gl='PILSVG',
                show_debug=False,
                screen_height=600,
                screen_width=600,
            )
            renderer.reset()

        observation, _ = env.reset()
        dones = {agent: False for agent in range(env.number_of_agents)}
        frames: List[np.ndarray] = []
        self._render_frame(renderer, show_render, frames)

        max_steps = config['learner_config'].get('max_steps_per_episode', 200)
        rewards_accum = {agent: 0.0 for agent in range(env.number_of_agents)}
        steps_executed = 0

        try:
            while steps_executed < max_steps and not all(dones.values()):
                obs_tensor = obs_dict_to_tensor(
                    observation=observation,
                    obs_type=env_config.observation_builder_config['type'],
                    n_agents=env.number_of_agents,
                    max_depth=env_config.observation_builder_config['max_depth'],
                    n_nodes=controller.config['n_nodes'],
                ).float()

                obs_tensor = normaliser.normalise(obs_tensor.unsqueeze(0)).squeeze(0)
                obs_tensor = obs_tensor.to(device)

                with torch.no_grad():
                    actions, _ = controller.select_action(obs_tensor)

                action_values = np.atleast_1d(actions.detach().cpu().numpy())
                action_dict = {
                    agent: None if dones[agent] else int(action_values[agent])
                    for agent in range(env.number_of_agents)
                }

                observation, rewards, dones, _ = env.step(action_dict)
                for agent_id, reward in rewards.items():
                    rewards_accum[agent_id] += reward

                self._render_frame(renderer, show_render, frames)
                steps_executed += 1
        finally:
            if renderer is not None:
                renderer.close_window()
            env.close()
            self._save_video(frames, video_path, fps=render_fps)

        total_return = sum(rewards_accum.values())
        print(f'Behaviour rollout finished after {steps_executed} steps, total return {total_return:.2f}.')
        self.assertGreater(steps_executed, 0, 'Behaviour rollout produced no environment steps.')
