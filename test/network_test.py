import sys
sys.path.append('../src')
import unittest
from argparse import Namespace
import torch
from src.networks.MLP import MLP
from src.networks.RNN import RNN, LSTM
from src.networks.CommNet import CommNet

class NetworkTest(unittest.TestCase):
    def test_MLP_reshaping(self):
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network = MLP(args, args.num_inputs)

        # Tensor case
        correct_tensor = torch.randn(args.batchsize, args.n_agents, args.num_inputs)
        output = network.adjust_input_dimensions(correct_tensor)
        self.assertTrue(isinstance(output, torch.Tensor))
        self.assertEqual(output.size(), (args.batchsize * args.n_agents, args.num_inputs))

        reshaped = network.adjust_output_dimensions(output)
        self.assertTrue(isinstance(reshaped, torch.Tensor))
        self.assertEqual(reshaped.size(), (args.batchsize, args.n_agents, args.num_inputs))
        
        # Tuple case
        correct_tuple = (correct_tensor, correct_tensor, correct_tensor)
        output = network.adjust_input_dimensions(correct_tuple)
        self.assertTrue(isinstance(output, tuple))
        self.assertEqual(output[0].size(), (args.batchsize * args.n_agents, args.num_inputs))
        self.assertEqual(output[1].size(), (args.batchsize * args.n_agents, args.num_inputs))  
        self.assertEqual(output[2].size(), (args.batchsize * args.n_agents, args.num_inputs))

        reshaped = network.adjust_output_dimensions(output)
        self.assertTrue(isinstance(reshaped, tuple))
        self.assertEqual(reshaped[0].size(), (args.batchsize, args.n_agents, args.num_inputs))
        self.assertEqual(reshaped[1].size(), (args.batchsize, args.n_agents, args.num_inputs))  
        self.assertEqual(reshaped[2].size(), (args.batchsize, args.n_agents, args.num_inputs))

        # Error Handling
        incorrect_tensor = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)
        incorrect_tuple = (incorrect_tensor, incorrect_tensor, incorrect_tensor)
        with self.assertRaises(ValueError):
            output = network.adjust_input_dimensions(incorrect_tensor)
            output = network.adjust_input_dimensions(incorrect_tuple)
        
    
    def test_MLP(self):
        """ Tests that the dimensions of the output Tensors are correct. """
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network = MLP(args, args.num_inputs)
        
        correct_tensor_1 = torch.randn(args.batchsize, args.num_inputs)
        actions, values = network(correct_tensor_1)
        self.assertEqual(actions.size(), (args.batchsize, args.n_actions))
        self.assertEqual(values.size(), (args.batchsize, 1))

        correct_tensor_2 = torch.randn(args.batchsize, args.n_agents, args.num_inputs)
        actions, values = network(correct_tensor_2)
        self.assertEqual(actions.size(), (args.batchsize, args.n_agents, args.n_actions))
        self.assertEqual(values.size(), (args.batchsize, args.n_agents, 1))

        incorrect_tensor = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)
        with self.assertRaises(ValueError):
            network(incorrect_tensor)


    def test_RNN(self):
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network: RNN = RNN(args, args.num_inputs)

        # Multi-Agent Forward Pass
        prev_hidden_state_2 = torch.randn(args.batchsize, args.n_agents, args.hid_size)
        correct_tensor_2 = torch.randn(args.batchsize, args.n_agents, args.num_inputs)
        actions, values, next_hidden_state = network(correct_tensor_2, prev_hidden_state_2)
        self.assertEqual(actions.size(), (args.batchsize, args.n_agents, args.n_actions))
        self.assertEqual(values.size(), (args.batchsize, args.n_agents, 1))
        self.assertEqual(next_hidden_state.size(), (args.batchsize, args.n_agents, args.hid_size))

        # Multi-Agent Target Network Forward Pass
        actions = network.forward_target_network(correct_tensor_2, prev_hidden_state_2)
        self.assertEqual(actions.size(), (args.batchsize, args.n_agents, args.n_actions))

        # Error Handling Forward Pass (Target and Local)
        incorrect_tensor_1 = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)
        with self.assertRaises(ValueError):
            network(incorrect_tensor_1, prev_hidden_state_2)
            network.forward_target_network(incorrect_tensor_1, prev_hidden_state_2)

        # Hidden State Initialiser
        self.assertEqual(network.init_hidden(args.batchsize).size(), (args.batchsize * args.n_agents, args.hid_size))

    
    def test_LSTM(self):
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network: LSTM = LSTM(args, args.num_inputs)

        # Hidden State Initialiser
        hidden_state_init, cell_state_init = network.init_hidden(args.batchsize)
        self.assertEqual(hidden_state_init.size(), (args.batchsize * args.n_agents, args.hid_size))
        self.assertEqual(cell_state_init.size(), hidden_state_init.size())

        # Multi-Agent Forward Pass
        prev_hidden_state_2 = torch.randn(args.batchsize, args.n_agents, args.hid_size)
        prev_cell_state_2 = prev_hidden_state_2.clone()
        correct_tensor_2 = torch.randn(args.batchsize, args.n_agents, args.num_inputs)
        actions, values, next_hidden_state, next_cell_state = network(correct_tensor_2, prev_hidden_state_2, prev_cell_state_2)
        self.assertEqual(actions.size(), (args.batchsize, args.n_agents, args.n_actions))
        self.assertEqual(values.size(), (args.batchsize, args.n_agents, 1))
        self.assertEqual(next_hidden_state.size(), (args.batchsize, args.n_agents, args.hid_size))
        self.assertEqual(next_cell_state.size(), next_hidden_state.size())

        # Multi-Agent Target Network Forward Pass
        actions = network.forward_target_network(correct_tensor_2, prev_hidden_state_2, prev_cell_state_2)
        self.assertEqual(actions.size(), (args.batchsize, args.n_agents, args.n_actions))

        # Error Handling Forward Pass (Target and Local)
        incorrect_tensor_1 = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)
        with self.assertRaises(ValueError):
            network(incorrect_tensor_1, prev_hidden_state_2, prev_cell_state_2)
            network.forward_target_network(incorrect_tensor_1, prev_hidden_state_2, prev_cell_state_2)


    def test_CommNet(self): 
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16, comm_passes = 3, recurrent=True)
        network = CommNet(args, args.num_inputs)

        # Hidden State Initialisation
        output = network.init_hidden(args.batchsize)
        self.assertEqual(output[0].size(), (args.batchsize * args.n_agents, args.hid_size))
        self.assertEqual(output[1].size(), output[0].size())

        # Forward Pass
        state = torch.randn(args.batchsize, args.n_agents, args.num_inputs)
        prev_hidden_state = torch.randn(args.batchsize, args.n_agents, args.hid_size)
        prev_cell_state = prev_hidden_state.clone()
        actions, value, next_hidden_state, next_cell_state = network(state, prev_hidden_state, prev_cell_state)
        self.assertEqual(actions.size(), (args.batchsize, args.n_agents, args.n_actions))
        self.assertEqual(value.size(), (args.batchsize, args.n_agents, 1))
        self.assertEqual(next_hidden_state.size(), (args.batchsize, args.n_agents, args.hid_size))
        self.assertEqual(next_cell_state.size(), next_hidden_state.size())

        # Error Handling
        incorrect_tensor = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)
        with self.assertRaises(ValueError):
            network(incorrect_tensor, prev_hidden_state, prev_cell_state)
            network(state, incorrect_tensor, prev_cell_state)
            network(state, prev_hidden_state, incorrect_tensor)

        # Forward Target Network
        actions = network.forward_target_network(state, prev_hidden_state)
        self.assertEqual(actions.size(), (args.batchsize, args.n_agents, args.n_actions))

        # Agent Mask
        info = {'comm_action': torch.randn(args.batchsize, args.n_agents, args.n_agents), 'alive_mask': torch.zeros(args.batchsize, args.n_agents)}
        num_agents_alive, agent_mask = network.get_agent_mask(args.batchsize, info)
        self.assertEqual(num_agents_alive, 0)
        self.assertEqual(agent_mask.size(), (args.batchsize, args.n_agents, args.n_agents))

        info = {'comm_action': torch.randn(args.batchsize, args.n_agents, args.n_agents), 'alive_mask': torch.ones(args.batchsize, args.n_agents)}
        num_agents_alive, agent_mask = network.get_agent_mask(args.batchsize, info)
        self.assertEqual(num_agents_alive, args.n_agents)


if __name__== '__main__':
    unittest.main()