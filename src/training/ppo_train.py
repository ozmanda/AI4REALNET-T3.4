import torch
import numpy as np
import torch.multiprocessing as mp
from typing import Dict

from src.configs.EnvConfig import FlatlandEnvConfig
from src.configs.ControllerConfigs import PPOControllerConfig
from src.configs.OptimiserConfig import AdamConfig


def train_ppo(experiment_path: str, n_workers: int):
    init_random_seeds()


def load_experiment_parameters(experiment_path: str) -> Dict:
    pass


def init_random_seeds() -> None:
    pass


if __name__ == '__main__':
    pass