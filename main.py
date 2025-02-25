from argparse import ArgumentParser, Namespace
from training.RNNTrainer import RNNTrainer
from training.CommNetTrainer import CommNetTrainer
from training.Trainer import Trainer
from networks.RNN import LSTM, RNN
from networks.CommNet import CommNet
from networks.MLP import MLP
from environments.env_small import small_flatland_env
from flatland.envs.rail_env import RailEnv
import time
from utils.utils import merge_dicts


if __name__ == '__main__': 
    parser = ArgumentParser()

    # Training Parameters
    parser.add_argument('--epochs', type=int, help='Number of epochs to train for', default=100)
    parser.add_argument('--epoch_size', type=int, help='Number of steps per epoch', default=10)
    parser.add_argument('--batch_size', type=int, help='Batch size for training', default=500)
    parser.add_argument('--max_steps', type=int, help='Maximum number of steps per episode', default=1000)

    # Environment Parameters
    parser.add_argument('--observation_type', type=str, help='Type of observation to use', default='tree')
    parser.add_argument('--max_tree_depth', type=int, help='Maximum depth of the tree observation if tree observation is used', default=3)

    # Network Parameters
    parser.add_argument('--comm', type=bool, help='Whether to use communication in the network', default=False)
    parser.add_argument('--hid_size', type=int, help='Size of the hidden layers', default=64)
    parser.add_argument('--recurrent', type=bool, help='Whether to use a recurrent network', default=True)
    parser.add_argument('--rnn_type', type=str, help='Type of RNN to use', default='lstm')

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


    env: RailEnv = small_flatland_env(observation=args.observation_type, max_tree_depth=args.max_tree_depth)
    _ = env.reset()
    args.n_agents = env.get_num_agents()
    args.n_actions = env.action_space[0]

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
        trainer = RNNTrainer(args, policynet, env)
    elif args.comm:
        policynet: CommNet = CommNet(args, obs_inputs)
        trainer = CommNetTrainer(args, policynet, env)
    else: 
        policynet = MLP(args, obs_inputs)
        trainer = Trainer(args, policynet, env)

    for epoch in range(args.epochs):

        epoch_start_time = time.time()
        epoch_stats = dict()

        for n in range(args.epoch_size):
            batch_info = trainer.train_batch()

        epoch_time = time.time() - epoch_start_time
        print(epoch_time)
        