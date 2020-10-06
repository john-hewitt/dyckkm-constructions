
import torch
from torch import nn

import utils
import math
import torch
import os
import torch.nn as nn
import yaml
import sys
import itertools
from argparse import ArgumentParser

import lstm_constructions
import simplernn_constructions

TANH_OF_1 = torch.tanh(torch.tensor(1.0))




class LSTMImpl(nn.Module):
  """
  Unbatched LSTM language model implementation with a focus on accessing intermediate reprs
  """
  def __init__(self, k, embedding_mtx,
      input_gate_hh_mtx, input_gate_hi_mtx, input_gate_bias,
      output_gate_hh_mtx, output_gate_hi_mtx, output_gate_bias,
      forget_gate_hh_mtx, forget_gate_hi_mtx, forget_gate_bias,
      new_cell_hh_mtx, new_cell_hi_mtx, new_cell_bias,
      softmax_mtx, softmax_bias
      ):
    super(LSTMImpl, self).__init__()
    self.embedding_mtx = nn.Embedding(embedding_mtx.shape[0], embedding_mtx.shape[1])
    self.embedding_mtx.weight = nn.Parameter(embedding_mtx)
    self.input_gate_hh_mtx=nn.Parameter(input_gate_hh_mtx)
    self.input_gate_hi_mtx=nn.Parameter(input_gate_hi_mtx)
    self.input_gate_bias=nn.Parameter(input_gate_bias)
    self.output_gate_hh_mtx=nn.Parameter(output_gate_hh_mtx)
    self.output_gate_hi_mtx=nn.Parameter(output_gate_hi_mtx)
    self.output_gate_bias=nn.Parameter(output_gate_bias)
    self.forget_gate_hh_mtx=nn.Parameter(forget_gate_hh_mtx)
    self.forget_gate_hi_mtx=nn.Parameter(forget_gate_hi_mtx)
    self.forget_gate_bias=nn.Parameter(forget_gate_bias)
    self.new_cell_hh_mtx=nn.Parameter(new_cell_hh_mtx)
    self.new_cell_hi_mtx=nn.Parameter(new_cell_hi_mtx)
    self.new_cell_bias=nn.Parameter(new_cell_bias)
    self.softmax_mtx = nn.Parameter(softmax_mtx)
    self.softmax_bias = nn.Parameter(softmax_bias)
    vocab = utils.get_vocab_of_bracket_types(k)[0]
    self.vocab = {x: c for c,x in enumerate(vocab)}

  def step(self, cell_state, hidden_state, symbol):
    """
    Compute a single recurrent update.

    Args:
      cell_state: old cell state c_{t-1}
      hidden_State: old hidden state h_{t-1}
      symbol: A string symbol from the vocabulary
    Returns:
      (new cell: c_t, new_hidden: h_t)
    """
    input_vec = self.embedding_mtx(torch.tensor(self.vocab[symbol]))

    input_gate = torch.sigmoid(torch.matmul(self.input_gate_hh_mtx, hidden_state) + torch.matmul(self.input_gate_hi_mtx, input_vec) + self.input_gate_bias)
    print('Input gate')
    print(input_gate)
    forget_gate = torch.sigmoid(torch.matmul(self.forget_gate_hh_mtx, hidden_state) + torch.matmul(self.forget_gate_hi_mtx, input_vec) + self.forget_gate_bias)
    print('Forget gate')
    print(forget_gate)
    output_gate = torch.sigmoid(torch.matmul(self.output_gate_hh_mtx, hidden_state) + torch.matmul(self.output_gate_hi_mtx, input_vec) + self.output_gate_bias)
    print('Output gate')
    print(output_gate)
    new_cell_candidate = torch.tanh(torch.matmul(self.new_cell_hh_mtx, hidden_state) + torch.matmul(self.new_cell_hi_mtx, input_vec) + self.new_cell_bias)
    print('New Cell Candidate')
    print(new_cell_candidate)

    new_cell = (forget_gate * cell_state) + (input_gate * new_cell_candidate)
    new_hidden = (output_gate * torch.tanh(new_cell))
    return new_cell, new_hidden

  def test_lm(self):
    hidden_state = torch.zeros(self.input_gate_hh_mtx.shape[0])
    cell_state = torch.zeros(self.input_gate_hh_mtx.shape[0])
    prefix = ''
    print('Accepting input')
    while True:
      symbol_input = input().strip()
      if symbol_input not in self.vocab:
        print('Invalid input')
        continue
      prefix += symbol_input
      print()
      print()
      print('prefix:', prefix)
      cell_state, hidden_state = self.step(cell_state, hidden_state, symbol_input)
      print('c', cell_state)
      print('h', hidden_state)
      probs = self.probs(hidden_state)
      print('probs:',{x: float(probs[self.vocab[x]].detach().numpy()) for x in self.vocab})
      print()


  def probs(self, hidden_state):
    """
    Computes the probability distribution over the next token given the current hidden state.
    """
    return torch.softmax(torch.matmul(self.softmax_mtx, hidden_state) + self.softmax_bias,0)

class SimpleRNNImpl(nn.Module):
  """
  Unbatched Simple RNN language model implementation with a focus on accessing intermediate reprs
  """
  def __init__(self, k, embedding_mtx,
      w_mtx, u_mtx, bias,
      softmax_mtx, softmax_bias
      ):
    super(SimpleRNNImpl, self).__init__()
    self.embedding_mtx = nn.Embedding(embedding_mtx.shape[0], embedding_mtx.shape[1])
    self.embedding_mtx.weight = nn.Parameter(embedding_mtx)
    self.w_mtx = w_mtx
    self.u_mtx = u_mtx
    self.bias = bias
    self.softmax_mtx = softmax_mtx
    self.softmax_bias = softmax_bias
    vocab = utils.get_vocab_of_bracket_types(k)[0]
    self.vocab = {x: c for c,x in enumerate(vocab)}

  def step(self, hidden_state, symbol):
    """
    Compute a single recurrent update.

    Args:
      cell_state: old cell state c_{t-1}
      hidden_State: old hidden state h_{t-1}
      symbol: A string symbol from the vocabulary
    Returns:
      (new cell: c_t, new_hidden: h_t)
    """
    input_vec = self.embedding_mtx(torch.tensor(self.vocab[symbol]))
    new_hidden = torch.sigmoid(torch.matmul(self.w_mtx, hidden_state) + torch.matmul(self.u_mtx, input_vec) + self.bias)
    return  new_hidden

  def probs(self, hidden_state):
    """
    Computes the probability distribution over the next token given the current hidden state.
    """
    #print('logits', torch.matmul(self.softmax_mtx, hidden_state) + self.softmax_bias)
    return torch.softmax(torch.matmul(self.softmax_mtx, hidden_state) + self.softmax_bias,0)
    
  def test_lm(self):
    hidden_state = torch.zeros(self.w_mtx.shape[0])
    prefix = ''
    print('Accepting input')
    while True:
      symbol_input = input().strip()
      if symbol_input not in self.vocab:
        print('Invalid input')
        continue
      print()
      print()
      prefix += symbol_input
      print('prefix:', prefix)
      hidden_state = self.step(hidden_state, symbol_input)
      print('h', hidden_state)
      probs = self.probs(hidden_state)
      print('probs:',{x: float(probs[self.vocab[x]].detach().numpy()) for x in self.vocab})
      print()

