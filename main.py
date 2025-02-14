from argparse import ArgumentParser, Namespace
from training.RNNTrainer import RNNTrainer
from training.CommNetTrainer import CommNetTrainer
from training.Trainer import Trainer
from networks.RNN import LSTM, RNN
from networks.CommNet import CommNet
from networks.MLP import MLP
from environments.env_small import small_flatland_env
from flatland.envs.rail_env import RailEnv


if __name__ == '__main_': 
    parser = ArgumentParser()
    parser.add_argument('--comm', type=bool, help='Whether to use communication in the network', default=False)
    parser.add_argument('--hid_size', type=int, help='Size of the hidden layers', default=64)
    parser.add_argument('--recurrent', type=bool, help='Whether to use a recurrent network', default=True)
    parser.add_argument('--rnn_type', type=str, help='Type of RNN to use', default='lstm')
    parser.add_argument('--detach_gap', type=int, help='Number of steps before detaching the hidden state', default=1)
    parser.add_argument('--observation_type', type=str, help='Type of observation to use', default='tree')
    parser.add_argument('--max_tree_depth', type=int, help='Maximum depth of the tree observation if tree observation is used', default=3)

    args: Namespace = parser.parse_args()


    env: RailEnv = small_flatland_env()
    if args.observation_type == 'tree':
        tree_nodes = int((4 ** (args.max_tree_depth + 1) - 1) / 3)   # geometric progression
        obs_features = 12
        obs_inputs = tree_nodes * obs_features
    elif args.observation_type == 'global':
        obs_inputs = env.observation_space[0].shape[0] * env.observation_space[0].shape[1] * 23
    else: 
        raise ValueError('Invalid observation type')


    if args.recurrent:
        policynet = LSTM(args, obs_inputs) if args.rnn_type == 'lstm' else RNN(args, obs_inputs)
        trainer = RNNTrainer()
    elif args.comm:
        policynet: CommNet = CommNet(args, obs_inputs)
        trainer = CommNetTrainer(args, policynet, env)
    else: 
        policynet = MLP(args, obs_inputs)
        trainer = Trainer(args, policynet, env)

    trainer.train_batch()