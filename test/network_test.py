import sys
sys.path.append('../src')
import unittest
from argparse import Namespace
import torch
from src.networks.MLP import MLP
from src.networks.RNN import RNN, LSTM

class NetworkTest(unittest.TestCase):
    def test_MLP_functions(self):
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network = MLP(args, args.num_inputs)

        # Tensor case
        correct_tensor = torch.randn(args.batchsize, args.n_agents, args.num_inputs)
        output = network.adjust_input_dimensions(correct_tensor)
        self.assertEqual(output.size(), (args.batchsize * args.n_agents, args.num_inputs))
        reshaped = network.adjust_output_dimensions(output)
        self.assertEqual(reshaped.size(), (args.batchsize, args.n_agents, args.num_inputs))
        
        # Tuple case
        correct_tuple = (correct_tensor, correct_tensor, correct_tensor)
        output = network.adjust_input_dimensions(correct_tuple)
        self.assertTrue(isinstance(output, tuple))
        self.assertEqual(output[0].size(), (args.batchsize * args.n_agents, args.num_inputs))
        self.assertEqual(output[1].size(), (args.batchsize * args.n_agents, args.num_inputs))  
        self.assertEqual(output[2].size(), (args.batchsize * args.n_agents, args.num_inputs))

        # Error Handling
        incorrect_tensor = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)
        incorrect_tuple = (incorrect_tensor, incorrect_tensor, incorrect_tensor)
        with self.assertRaises(ValueError):
            output = network.adjust_input_dimensions(incorrect_tensor)
            output = network.adjust_input_dimensions(incorrect_tuple)

        
    
    def test_MLP_dimensions(self):
        """ Tests that the dimensions of the output Tensors are correct. """
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network = MLP(args, args.num_inputs)
        
        correct_tensor_1 = torch.randn(args.batchsize, args.num_inputs)
        actions, values = network(correct_tensor_1)
        assert actions.size == (args.batchsize, args.n_agents, args.n_actions)
        assert values.size == (args.batchsize, args.n_agents, 1)

        correct_tensor_2 = torch.randn(args.batchsize, args.n_agents, args.num_inputs)
        actions, values = network(correct_tensor_2)
        assert actions.size == (args.batchsize, args.n_actions)
        assert values.size == (args.batchsize, 1)

        incorrect_tensor = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)
        with self.assertRaises(ValueError):
            network(incorrect_tensor)


    def test_RNN_dimensions(self):
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network: RNN = RNN(args, args.num_inputs)

        # Single-Agent Case
        prev_hidden_state_1 = torch.randn(args.batchsize, args.hid_size)
        correct_tensor_1 = torch.randn(args.batchsize, args.num_inputs)
        actions, values, next_hidden_state = network(correct_tensor_1, prev_hidden_state_1)
        self.assertEqual(actions.size(), (args.batchsize, args.n_actions))
        self.assertEqual(values.size(), (args.batchsize, 1))
        self.assertEqual(next_hidden_state.size(), (args.batchsize, args.hid_size))

        # Multi-Agent Case
        prev_hidden_state_2 = torch.randn(args.batchsize, args.n_agents, args.hid_size)
        correct_tensor_2 = torch.randn(args.batchsize, args.n_agents, args.num_inputs)
        actions, values, next_hidden_state = network(correct_tensor_2, prev_hidden_state_2)
        self.assertEqual(actions.size(), (args.batchsize * args.n_agents, args.n_actions))
        self.assertEqual(values.size(), (args.batchsize * args.n_agents, 1))
        self.assertEqual(next_hidden_state.size(), (args.batchsize * args.n_agents, args.hid_size))

        # Error Handling 
        incorrect_tensor_1 = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)
        incorrect_tensor_2 = torch.randn(args.batchsize, args.num_inputs, 4)
        with self.assertRaises(ValueError):
            output = network(incorrect_tensor_1, prev_hidden_state_1)

if __name__== '__main__':
    unittest.main()