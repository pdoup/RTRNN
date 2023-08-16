# -*- coding: utf-8 -*-
"""
Created on Fri Aug  4 20:32:33 2023.

@author: Panagiotis Doupidis
"""
import torch
import inspect
from typing import List
from torch import nn
from utils import get_params


class RTRNNBlock(nn.Module):
    """
    RTRNNBlock class is a PyTorch implementation of a Randomized Temporal RNN.
    (RTRNNBlock) with skip connections in  its fully connected network (FCN).
    It takes several arguments during initialization such as input_size,
    hidden_size, output_size, num_layers, dropout_prob, window_size, and device.
    These arguments are used to configure various aspects of the RTRNNBlock model
    such as its architecture and training parameters.
    """

    def __init__(self, input_size, hidden_size, output_size, num_layers=1,
                 dropout_prob=0.1, window_size=10, device='cpu'):
        """
        A PyTorch implementation of a Randomized Temporal Recurrent Neural
        Network (RTRNNBlock) with skip connections in the FCN.

        Parameters
        ----------
        input_size : int
            The number of expected features in the input.
        hidden_size : int
            The number of features in the hidden state.
        output_size : int
            The number of expected features in the output.
        num_layers : int, optional
            The number of recurrent layers. Defaults to 1.
        dropout_prob : float, optional
            The probability of an element to be zeroed. Defaults to 0.1.
        window_size : int, optional
            The window size to use for shuffling the input. Defaults to 10.
        """

        super(RTRNNBlock, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        self.device = device
        self.num_rnns = window_size // 2 if window_size % 2 == 0 else (
            window_size + 1) // 2
        self.params = get_params(inspect.currentframe())
        self.rnns = nn.ModuleList([nn.RNN(input_size, hidden_size, num_layers,
                                          nonlinearity='relu') for _ in
                                   range(self.num_rnns)])
        self.rnn_orig = nn.RNN(input_size, hidden_size,
                               num_layers, nonlinearity='tanh')
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size * 16)
        self.dropout1 = nn.Dropout(dropout_prob)
        self.fc2 = nn.Linear(hidden_size * 16, hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc3 = nn.Linear(hidden_size * 17, output_size)

    def forward(self, x):
        """
        Forward pass through the RTRNNBlock model.

        Parameters
        ----------
        x : torch.Tensor
            The input data.

        Returns
        -------
        torch.Tensor
            The output of the RTRNNBlock model.
        """
        # Pass the shuffled input through the RNN layers and add their outputs
        h0s = [torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(
            self.device) for _ in range(self.num_rnns)]
        out_rs = [rnn(x[:, torch.randperm(x.size(1)), :], h0)[0][:, -1, :]
                  for rnn, h0 in zip(self.rnns, h0s)]
        out_r_sum = torch.stack(out_rs).sum(dim=0)

        # Pass the original input through the additional RNN layer
        h0_orig = torch.zeros(self.num_layers, x.size(1),
                              self.hidden_size).to(self.device)
        out_r_orig, _ = self.rnn_orig(x, h0_orig)
        out_r_orig = out_r_orig[:, -1, :]

        # Concatenate the sum of the outputs of the RNNs in the list with the
        # output from the single RNN
        out_r = torch.cat([out_r_sum, out_r_orig], dim=1)

        # Pass the concatenated outputs through the fully connected and dropout
        # layers
        out1 = self.fc1(out_r)
        out1 = torch.nn.functional.rrelu(out1)
        out1 = self.dropout1(out1)
        out2 = self.fc2(out1)
        out2 = torch.tanh(out2)
        out2 = self.dropout2(out2)
        out3 = torch.cat([out1, out2], dim=1)
        out3 = self.fc3(out3)

        return out3


class FusedRTRNN(nn.Module):
    """Fused RTRNN implementation."""

    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int,
                 num_layers: List[int], num_blocks: int, dropout_prob: float = 0.1,
                 window_size: int = 10, device: str = 'cpu'):
        """
        A PyTorch module that implements a fused RTRNN model.

        This model consists of multiple RTRNN blocks, each with its own hidden size
        and number of layers. The outputs of these blocks are merged using a
        weighted average reduction strategy to produce the final output.

        Parameters
        ----------
        input_size : int
            The size of the input features.
        hidden_sizes : List[int]
            A list of hidden sizes for each RTRNN block.
        output_size : int
            The size of the output.
        num_layers : List[int]
            A list of the number of layers for each RTRNN block.
        dropout_prob : float, optional
            The probability of dropout. Defaults to 0.1.
        window_size : int, optional
            The size of the window for the RTRNN blocks. Defaults to 10.
        num_blocks : int, optional
            The number of RTRNN blocks in the model. Defaults to 2.

        Raises
        ------
        ValueError
            If the lengths of hidden_sizes and num_layers do not match the number
            of RTRNN blocks specified by num_blocks.

        Attributes
        ----------
        num_blocks : int
            The number of RTRNN blocks in the model.
        rtrnns : nn.ModuleList
            A list of RTRNNBlock modules, one for each block in the model.
        weights : List[float]
            A list of weights used to compute the weighted average of the outputs
            from each RTRNN block.
        """
        super(FusedRTRNN, self).__init__()
        self.num_blocks = num_blocks
        self.window_size = window_size
        if len(hidden_sizes) != num_blocks or len(num_layers) != num_blocks:
            raise ValueError(
                "The lengths of hidden_sizes and num_layers must match the number of RTRNN blocks specified by num_blocks")
        self.params = get_params(inspect.currentframe())
        self.rtrnns = nn.ModuleList([
            RTRNNBlock(input_size, hidden_sizes[i], output_size, num_layers[i],
                       dropout_prob, window_size, device)
            for i in range(num_blocks)])

        self.weights = [hidden_size / sum(hidden_sizes)
                        for hidden_size in hidden_sizes]

    def forward(self, x):
        """
        Forward pass through the FusedRTRNN model.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor with shape (batch_size, sequence_length, input_size).

        Returns
        -------
        torch.Tensor
            The output tensor with shape (batch_size, sequence_length, output_size).
        """
        out_ts = [rtrnn(x) * weight for rtrnn,
                  weight in zip(self.rtrnns, self.weights)]
        out_t_avg = torch.stack(out_ts).sum(dim=0)

        return out_t_avg
