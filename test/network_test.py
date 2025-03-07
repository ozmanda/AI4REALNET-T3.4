import unittest
from argparse import Namespace
import torch
from networks.MLP import MLP
from networks.RNN import RNN, LSTM
import sys
sys.path.append('../networks')

class NetworkTest(unittest.TestCase):
    
    def test_MLP_dimensions(self):
        """ Tests that the dimensions of the output Tensors are correct. """
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network = MLP(args, args.num_inputs)
        
        correct_tensor_1 = torch.randn(args.batchsize, args.nagents, args.nfeatures)
        correct_tensor_2 = torch.randn(args.batchsize, args.nfeatures)
        incorrect_tensor = torch.randn(args.batchsize, args.nagents, args.nfeatures, 4)

        actions, values = network(correct_tensor_1)
        assert actions.size == (args.batchsize, args.n_agents, args.n_actions)
        assert values.size == (args.batchsize, args.n_agents, 1)

        actions, values = network(correct_tensor_2)
        assert actions.size == (args.batchsize, args.n_actions)
        assert values.size == (args.batchsize, 1)

        with self.assertRaises(ValueError):
            network(incorrect_tensor)

# if __name__== '__main__':
#     unittest.main()