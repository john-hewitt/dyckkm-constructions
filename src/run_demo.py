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

import hand_rnn
import lstm_constructions
import simplernn_constructions


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('rnn_type', type=str, default='lstm', help='Whether to use a Simple RNN (simplernn) or LSTM (lstm)')
  parser.add_argument('encoding_type', type=str, default='log', help='Whether to use the O(mk) solution (linear) or the O(m\log k) solution (log)')
  parser.add_argument('k', type=int, default=8, help='Vocabulary size. Must be a power of two, >1 for this implementation.')
  parser.add_argument('m', type=int, default=4, help='Maximum nesting depth.')
  args = parser.parse_args()
  print('running with commands {}'.format(args))
  k = args.k
  m = args.m
  if args.rnn_type == 'lstm':
    if args.encoding_type == 'log':
      params = lstm_constructions.get_dyckkm_lstm_mlogk_params(k,m)
    else:
      params = lstm_constructions.get_dyckkm_lstm_mk_params(k,m)
    lm = hand_rnn.LSTMImpl(k, *params)
  elif args.rnn_type =='simplernn':
    if args.encoding_type == 'log':
      params = simplernn_constructions.get_dyckkm_simplernn_mlogk_params(k,m)
    else:
      params = simplernn_constructions.get_dyckkm_simplernn_mk_params(k,m)
    lm = hand_rnn.SimpleRNNImpl(k, *params)
  lm.test_lm()
