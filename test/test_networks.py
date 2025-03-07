"""
Tests the functionality of all network modules, ensuring that the proper dimensions are output and that the forward pass doesn't return NaNs for a proper input.
"""
import sys
sys.path.append('../networks')
import pytest

import torch
from argparse import Namespace
from networks.MLP import MLP
from networks.RNN import RNN, LSTM


def test_MLP_dimensions():
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

    with pytest.raises(ValueError):
        output = network(incorrect_tensor)


# def test_RNN_dimensions():
#     args = Namespace(hid_size=128, n_actions=5, batchsize = 10, n_agents = 8, num_inputs = 16)
#     network = RNN(args, args.num_inputs)

#     prev_hidden_state_1 = torch.randn(args.batchsize, args.n_agents, args.hid_size)
#     correct_tensor_1 = torch.randn(args.batchsize, args.n_agents, args.num_inputs)

#     prev_hidden_state_2 = torch.randn(args.batchsize, args.hid_size)
#     correct_tensor_2 = torch.randn(args.batchsize, args.num_inputs)

#     incorrect_tensor = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)
#     incorrect_tensor = torch.randn(args.batchsize, args.n_agents, args.num_inputs, 4)

#     actions, values, next_hidden_state = network(correct_tensor_1, prev_hidden_state_1)
#     assert actions.size == (args.batchsize, args.n_agents, args.n_actions)
#     assert values.size == (args.batchsize, args.n_agents, 1)
#     assert next_hidden_state.size == (args.batchsize, args.n_agents, args.hid_size)

#     actions, values, next_hidden_state = network(correct_tensor_2, prev_hidden_state_2)
#     assert actions.size == (args.batchsize, args.n_actions)
#     assert values.size == (args.batchsize, 1)
#     assert next_hidden_state.size == (args.batchsize, args.hid_size)

#     with pytest.raises(ValueError):
#         output = network(incorrect_tensor, prev_hidden_state_1)
    


