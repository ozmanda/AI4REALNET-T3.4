import os
import time
import torch
import numpy as np
from typing import Dict, NamedTuple
from argparse import ArgumentParser, Namespace

from src.training.Trainer import Trainer
from src.training.RNNTrainer import RNNTrainer
from src.training.CommNetTrainer import CommNetTrainer

from src.networks.MLP import MLP
from src.networks.RNN import LSTM, RNN
from src.networks.CommNet import CommNet

from flatland.envs.rail_env import RailEnv
from src.environments.env_small import small_flatland_env

from src.utils.utils import merge_dicts
from src.utils.log_utils import init_logger


if __name__ == '__main__': 
    parser = ArgumentParser()

    # Training Parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for', default=100)
    parser.add_argument('--epoch_size', type=int, help='Number of steps per epoch', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=500)
    parser.add_argument('--max_steps', type=int, help='Maximum number of steps per episode', default=1000)

    # Model Saving
    parser.add_argument('--save', type=bool, help='Whether to save the model', default=True)
    parser.add_argument('--save_whole_model', type=bool, help='Whether to save the whole model or just the state dict', default=False)
    parser.add_argument('--save_dir', type=str, help='Directory to save models', default='models/')
    parser.add_argument('--save_every', type=int, help='Frequency of saving models', default=10)
    parser.add_argument('--model_name', type=str, help='Name of the model to save without an extension', default=None)

    # Model Loading
    parser.add_argument('--load', type=str, help='Model to load', default=False)
    parser.add_argument('--load_dir', type=str, help='Directory to load models from', default='models/')

    # Log Settings
    parser.add_argument('--log', type=bool, help='Whether to log the training', default=True)
    parser.add_argument('--log_dir', type=str, help='Directory to save logs, default is modeldir', default=None)

    # Environment Parameters
    parser.add_argument('--observation_type', type=str, help='Type of observation to use', default='tree')
    parser.add_argument('--max_tree_depth', type=int, help='Maximum depth of the tree observation if tree observation is used', default=3)

    # Network Parameters
    parser.add_argument('--hid_size', type=int, help='Size of the hidden layers', default=64)
    parser.add_argument('--recurrent', type=bool, help='Whether to use a recurrent network', default=True)
    parser.add_argument('--rnn_type', type=str, help='Type of RNN to use', default='lstm')

    parser.add_argument('--comm', type=bool, help='Whether to use communication in the network', default=False)
    parser.add_argument('--comm_passes', type=int, help='Number of passes through the communication network', default=1)
    # Learning Parameters
    parser.add_argument('--learning_rate', type=float, help='Learning rate for the optimizer', default=1e-2)
    parser.add_argument('--detach_gap', type=int, help='Number of steps before detaching the hidden state', default=1)
    parser.add_argument('--normalise_rewards', type=bool, help='Whether to normalise rewards', default=True)
    parser.add_argument('--alpha', type=float, help='Alpha value for the RMSProp optimisation algorithm', default=0.99)
    parser.add_argument('--epsilon', type=float, help='Epsilon value for the CLIP_loss', default=0.2)
    parser.add_argument('--eps', type=float, help='Epsilon value for the RMSProp optimisation algorithm', default=1e-8)
    parser.add_argument('--value_coefficient', type=float, help='Coefficient for the value loss', default=1)
    parser.add_argument('--entropy_coefficient', type=float, help='Coefficient for the entropy loss', default=0.001)
    parser.add_argument('--normalise_advantage', type=bool, help='Whether to normalise the advantage', default=True)
    parser.add_argument('--gamma', type=float, help='Discount factor for rewards', default=0.99)
    parser.add_argument('--target_update_freq', type=int, help='Frequency of updating the target network', default=100)

    args: Namespace = parser.parse_args()

    # Environment initialisation 
    env: RailEnv = small_flatland_env(observation=args.observation_type, max_tree_depth=args.max_tree_depth)
    _ = env.reset()
    args.n_agents = env.get_num_agents()
    args.n_actions = env.action_space[0]

    # Observation settings
    if args.observation_type == 'tree':
        tree_nodes = int((4 ** (args.max_tree_depth + 1) - 1) / 3)   # geometric progression
        obs_features = 12
        obs_inputs = tree_nodes * obs_features
    elif args.observation_type == 'global':
        obs_inputs = env.observation_space[0].shape[0] * env.observation_space[0].shape[1] * 23
    else: 
        raise ValueError('Invalid observation type')
    
    # model saving settings
    if args.save:
        if not args.model_name:
            args.model_name = f'model_{time.strftime("%d%m%y_%H%M")}'
        args.save_dir = os.path.join(args.save_dir, args.model_name)
        if not os.path.isdir(args.save_dir):
            os.makedirs(args.save_dir)

    # logger initialisation
    if args.log:
        log: Dict[str, NamedTuple] = init_logger()
        if not args.log_dir:
            args.log_dir = args.save_dir

    # Network initialisation
    if args.recurrent:
        policynet = LSTM(args, obs_inputs) if args.rnn_type == 'lstm' else RNN(args, obs_inputs)
        trainer = RNNTrainer(args, policynet, env)
    elif args.comm:
        policynet: CommNet = CommNet(args, obs_inputs)
        trainer = CommNetTrainer(args, policynet, env)
    else: 
        policynet = MLP(args, obs_inputs)
        trainer = Trainer(args, policynet, env)
    
    # load most recent saved model
    if args.load:
        model_files = [f for f in os.listdir(args.load_dir) if os.path.isfile(os.path.join(args.load_dir, f))]
        if model_files:
            latest_model = max(model_files, key=lambda x: os.path.getctime(os.path.join(args.load_dir, x)))
            args.load = os.path.join(args.load_dir, latest_model)
        else:
            raise FileNotFoundError(f"No model files found in directory {args.load_dir}")
        
        policynet.load_state_dict(torch.load(args.load))

    #? move this into the trainer class?
    # TODO: more printouts to know where code is running / stuck
    # Main epoch training loop
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        epoch_stats = dict()

        for n in range(args.epoch_size):
            batch_info = trainer.train_batch()

        epoch_time = time.time() - epoch_start_time
        print(f'Epoch {epoch} \tReward {np.sum(batch_info["sum_reward"])} \tTime {epoch_time}')

        if args.save:
            if epoch % args.save_every == 0: 
                if args.save_whole_model:
                    torch.save(policynet, os.path.join(args.save_dir, f'{args.model_name}_epoch_{epoch}.pt'))
                else:
                    torch.save(policynet.state_dict(), os.path.join(args.save_dir, f'{args.model_name}_state_dict_epoch_{epoch}.pt'))
        