import sys
sys.path.append('../src')
import unittest
from argparse import Namespace
import torch
import numpy as np
from typing import Union
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
        self.network_test(network, args.batchsize)


    def test_RNN(self):
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network: RNN = RNN(args, args.num_inputs)
        self.network_test(network, args.batchsize)

    
    def test_LSTM(self):
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
        network: LSTM = LSTM(args, args.num_inputs)
        self.network_test(network, args.batchsize)


    def test_CommNet(self):
        """
        Tests the functionalities of the CommNet network. Three main factors influence the network that need to be tested: recurrency, shared weights, the number of communication passes (comm_passes) and weight initialisation (comm_init). 
            - recurrent         {True, False}
            - share_weights     {True, False}
            - comm_passes       {1, 4}
            - comm_init         {'zeros', 'rand'}

        Both CommNet types (recurrent and non-recurrent) can be tested as LSTM networks, as they both take hidden and cell state inputs.
        """
        # CommNet Variant 1: No Recurrent, No Shared Weights, Random Weight Initialisation
        args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16, comm_passes = 1, recurrent=False, share_weights = False, comm_init = 'rand', hard_attention = True)
        network = CommNet(args, args.num_inputs)
        self.network_test(network, args.batchsize)

        # CommNet Variant 2: Recurrent, Shared Weights, Zero Weight Initialisation
        args.recurrent = True
        args.share_weights = True
        args.comm_passes = 4
        args.comm_init = 'zeros'
        network = CommNet(args, args.num_inputs)
        self.network_test(network, args.batchsize)


    def network_test(self, network: Union[CommNet, LSTM, RNN, MLP], batchsize: int):
        
        self.forward_pass_test(network, batchsize)
        self.error_handling_test(network, batchsize)

        if isinstance(network, RNN) or isinstance(network, LSTM) or isinstance(network, CommNet):
            self.init_test(network, batchsize)
            self.forward_target_test(network, batchsize)
        
        if isinstance(network, CommNet):
            self.agent_mask_test(network, batchsize)
        
    
    def init_test(self, network: Union[CommNet, LSTM, RNN], batchsize: int):
        output = network.init_hidden(batchsize)
        if isinstance(network, RNN): 
            self.assertEqual(output.size(), (batchsize * network.n_agents, network.hid_size))
        if isinstance(network, LSTM) or isinstance(network, CommNet):
            self.assertEqual(output[0].size(), (batchsize * network.n_agents, network.hid_size))
            self.assertEqual(output[1].size(), output[0].size())


    def agent_mask_test(self, network: CommNet, batchsize): 
        info = {'comm_action': torch.randn(batchsize, network.n_agents, network.n_agents), 'alive_mask': torch.zeros(batchsize, network.n_agents)}
        num_agents_alive, agent_mask = network._get_agent_mask(batchsize, info)
        self.assertTrue(torch.equal(num_agents_alive, torch.zeros(batchsize)))
        self.assertEqual(agent_mask.size(), (batchsize, network.n_agents, network.n_agents))

        info = {'comm_action': torch.randn(batchsize, network.n_agents, network.n_agents), 'alive_mask': torch.ones(batchsize, network.n_agents)}
        num_agents_alive, agent_mask = network._get_agent_mask(batchsize, info)
        self.assertTrue(torch.equal(num_agents_alive, torch.full(batchsize, network.n_agents)))


    def error_handling_test(self, network: Union[CommNet, LSTM, RNN, MLP], batchsize: int): 
        correct_state = torch.randn(batchsize, network.n_agents, network.n_features)
        hidden_state = torch.randn(batchsize, network.n_agents, network.hid_size)
        incorrect_tensor = torch.randn(batchsize, network.n_agents, network.n_features, 4)
        
        if isinstance(network, LSTM) or isinstance(network, CommNet): 
            with self.assertRaises(ValueError): 
                network(incorrect_tensor, hidden_state, hidden_state.clone())
                network(correct_state, incorrect_tensor, hidden_state.clone())
                network(correct_state, hidden_state, incorrect_tensor)

        elif isinstance(network, RNN):
            with self.assertRaises(ValueError): 
                network(incorrect_tensor, hidden_state)
                network(correct_state, incorrect_tensor)

        else: 
            with self.assertRaises(ValueError): 
                network(incorrect_tensor)


    def forward_pass_test(self, network: Union[CommNet, LSTM, RNN, MLP], batchsize: int):
        correct_state = torch.randn(batchsize, network.n_agents, network.n_features)
        hidden_state = torch.randn(batchsize, network.n_agents, network.hid_size)

        if isinstance(network, LSTM): 
            actions, value, next_hidden_state, next_cell_state = network(correct_state, hidden_state, hidden_state.clone())
            self.assertEqual(next_hidden_state.size(), (batchsize, network.n_agents, network.hid_size))
            self.assertEqual(next_cell_state.size(), next_hidden_state.size())
            
        if isinstance(network, CommNet): 
            info = {'comm_action': torch.randn(batchsize, network.n_agents, network.n_agents), 'alive_mask': torch.zeros(batchsize, network.n_agents)}
            actions, value, next_hidden_state, next_cell_state = network(correct_state, hidden_state, hidden_state.clone(), info)
            if network.recurrent: 
                self.assertEqual(next_hidden_state.size(), (batchsize, network.n_agents, network.hid_size))
                self.assertEqual(next_cell_state.size(), next_hidden_state.size())

        elif isinstance(network, RNN):
            actions, value, next_hidden_state = network(correct_state, hidden_state)
            self.assertEqual(next_hidden_state.size(), (batchsize, network.n_agents, network.hid_size))

        else: 
            actions, value = network(correct_state)
        
        self.assertEqual(actions.size(), (batchsize, network.n_agents, network.n_actions))
        self.assertEqual(value.size(), (batchsize, network.n_agents, 1))


    def forward_target_test(self, network: Union[CommNet, LSTM, RNN, MLP], batchsize: int): 
        correct_state = torch.randn(batchsize, network.n_agents, network.n_features)
        hidden_state = torch.randn(batchsize, network.n_agents, network.hid_size)

        if isinstance(network, LSTM): 
            actions = network.forward_target_network(correct_state, hidden_state, hidden_state.clone())
            self.assertEqual(actions.size(), (batchsize, network.n_agents, network.n_actions))

        elif isinstance(network, CommNet):
            info = {'comm_action': torch.rand(batchsize, network.n_agents, network.n_agents), 'alive_mask': torch.zeros(batchsize, network.n_agents)}
            actions = network.forward_target_network(correct_state, hidden_state, hidden_state.clone(), info=info)
            self.assertEqual(actions.size(), (batchsize, network.n_agents, network.n_actions))

        elif isinstance(network, RNN):
            actions = network.forward_target_network(correct_state, hidden_state)
            self.assertEqual(actions.size(), (batchsize, network.n_agents, network.n_actions))


if __name__== '__main__':
    unittest.main()