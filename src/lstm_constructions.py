"""
Contains analytic constructions of LSTMs
that generate the m-bounded Dyck-k languages
"""

import utils
import torch
import itertools
from torch import nn
import math


TANH_OF_1 = torch.tanh(torch.tensor(1.0))
SCALE_ALPHA = 1e4

def get_efficient_ctilde_u_mtx(k):
  """
  Constructs the efficient O(\log k)$ encoding
  """
  embeddings = []
  softmaxes = []
  logk = int(math.log2(k))
  for i in range(k):
    all_zeros_1 = torch.zeros(logk)
    bitlist = [int(x) for x in list(bin(i)[2:])]
    for j in range(len(bitlist)):
      all_zeros_1[-(j+1)] = float(bitlist[-(j+1)])
    all_zeros_2 = 1-all_zeros_1
    all_neg_ones = -1*torch.ones(logk - 1)
    vec = torch.cat((all_zeros_1, all_zeros_2, all_neg_ones))
    softmax_vec = torch.cat((all_zeros_1, all_zeros_2, -all_neg_ones))
    #print('Symbol:', i, 'encoding', vec, 'Sum of elts', sum(vec))
    embeddings.append(vec)
    softmaxes.append(softmax_vec)
  embeddings = torch.stack(embeddings)
  #print(embeddings.shape)
  embeddings = torch.cat((embeddings, torch.zeros(k, 3*logk-1), torch.zeros(1, 3*logk-1)),0)
  softmaxes = torch.stack(softmaxes)
  #print('emb',embeddings.shape, 3*(logk)-1,2*k+1)
  #print(embeddings)
  return embeddings.transpose(0,1), softmaxes




def get_dyckkm_lstm_mlogk_params(k=4,m=3):
  """
  Returns LSTM parameters for O(m\log k) construction
  generating Dyck-(k,m)
  """
  assert math.log2(k).is_integer()
  vocab = utils.get_vocab_of_bracket_types(k)[0]
  vocab_size = len(vocab)
  open_bracket_indices = list(range(0,k))
  close_bracket_indices = list(range(k,2*k))
  end_symbol_index = 2*k
  slot_size = 3*(int(math.log2(k)))-1
  print('Slot size', slot_size)
  hidden_size = slot_size*m
  print('hidden size', hidden_size)
  num_stack_states = m

  # Vocabulary
  embedding_weights = torch.eye(vocab_size, vocab_size)
  #print(embedding_weights, embedding_weights.shape)

  # Input gate
  input_gate_hi_mtx = torch.stack(tuple((sum([embedding_weights[i] for i in open_bracket_indices]) for _ in range(hidden_size))))
  input_gate_hh_rows = []
  input_gate_bias_vals = []
  for i in range(m):
    if i == 0:
      neg_ones = -1*torch.ones(slot_size*num_stack_states)
      row = neg_ones
    else:
      ones = torch.ones(slot_size)
      neg_ones = -1*ones
      zeros_initial = torch.zeros(max(slot_size*(i-1),0))
      zeros_final = torch.zeros(max(slot_size*(num_stack_states-2-max(i-1,0)),0))
      row = torch.cat((zeros_initial, ones, neg_ones, zeros_final))
    input_gate_hh_rows.extend([row for _ in range(slot_size)])
  #for thing in input_gate_hh_rows:
  #  print(len(thing), thing)
  #print([len(x) for x in input_gate_hh_rows])
  input_gate_hh_mtx = torch.stack(input_gate_hh_rows)
  input_gate_bias = torch.tensor([-0.5 if i < slot_size else -1.5 for i in range(hidden_size)])
  print('input')
  print('b_i', input_gate_bias)
  print('W_i', input_gate_hh_mtx)
  print('U_i', input_gate_hi_mtx)

  # Forget gate
  forget_gate_hi_mtx = -torch.stack(tuple((sum([embedding_weights[i] for i in close_bracket_indices]) for _ in range(hidden_size))))
  forget_gate_hh_rows = []
  for i in range(num_stack_states):
    neg_ones = -1*torch.ones(slot_size)
    zeros_initial = torch.zeros(slot_size*i)
    zeros_final = torch.zeros(max(slot_size*(num_stack_states-1-i), 0))
    row = torch.cat((zeros_initial, neg_ones, zeros_final))
    forget_gate_hh_rows.extend([row for _ in range(slot_size)])
  forget_gate_hh_mtx = torch.stack(forget_gate_hh_rows)
  forget_gate_bias = torch.ones(hidden_size)*0.5*TANH_OF_1 + 1
  print('forget')
  print('b_f', forget_gate_bias)
  print('W_f', forget_gate_hh_mtx)
  print('U_f', forget_gate_hi_mtx)

  # Output gate
  output_gate_hi_mtx = torch.stack(tuple((sum((embedding_weights[i] for i in close_bracket_indices)) for _ in range(hidden_size))))
  output_gate_hh_rows = []
  for i in range(num_stack_states):
    neg_max = -num_stack_states*torch.ones(slot_size*max(i-1, 0))
    neg_ones = -torch.ones(slot_size*min(i+1,2))
    zeros = torch.zeros(slot_size*(num_stack_states-i-1))
    row = torch.cat((neg_max, neg_ones, zeros))
    output_gate_hh_rows.extend([row for _ in range(slot_size)])
  #for row in output_gate_hh_rows:
  #  print(len(row), row)
  output_gate_hh_mtx = torch.stack(output_gate_hh_rows).transpose(0,1)
  output_gate_bias = 0.5*torch.ones(hidden_size)
  print('output')
  print('b_o', output_gate_bias)
  print('W_o', output_gate_hh_mtx)
  print('U_o', output_gate_hi_mtx)

  # New cell candidate
  #new_cell_hi_mtx = torch.stack(tuple(itertools.chain.from_iterable((embedding_weights[i] for i in open_bracket_indices) for x in range(num_stack_states))))
  efficient_embeddings, efficient_softmaxes = get_efficient_ctilde_u_mtx(k)
  new_cell_hi_mtx = torch.stack(tuple(itertools.chain.from_iterable(efficient_embeddings for x in range(num_stack_states))))
  new_cell_hh_mtx = torch.zeros(hidden_size, hidden_size)
  new_cell_bias = torch.zeros(hidden_size)
  print('new cell')
  print('b_\\tilde{c}', new_cell_bias, new_cell_bias.shape)
  print('W_\\tilde{c}', new_cell_hh_mtx, new_cell_hh_mtx.shape)
  print('U_\\tilde{c}', new_cell_hi_mtx, new_cell_hi_mtx.shape)

  # Softmax
  softmax_mtx = torch.zeros(len(vocab), hidden_size).float()
  softmax_bias = torch.zeros(len(vocab))

  # close-i
  for i,index_of_i in enumerate(close_bracket_indices):
    for j in range(num_stack_states):
      #softmax_mtx[index_of_i,k*j+ i] = 1
      softmax_mtx[index_of_i,slot_size*j:slot_size*(j+1)] = efficient_softmaxes[i]
    softmax_bias[index_of_i] = -TANH_OF_1*0.5
  # open-i
  for i, index_of_i in enumerate(open_bracket_indices):
    for j in range(slot_size):
      softmax_mtx[index_of_i,slot_size*(m-1)+j] = -1
    softmax_bias[index_of_i] = 0.5*TANH_OF_1
  # end
  for i in range(slot_size):
    for j in range(num_stack_states):
      softmax_mtx[end_symbol_index,slot_size*j+i] = -1
  softmax_bias[end_symbol_index]= 0.5*TANH_OF_1
  print('softmax')
  print('softmax_mtx', softmax_mtx)
  print('softmax_bias', softmax_bias)

  return (embedding_weights, 1e4*input_gate_hh_mtx, 1e4*input_gate_hi_mtx, 1e4*input_gate_bias,
      1e4*output_gate_hh_mtx, 1e4*output_gate_hi_mtx, 1e4*output_gate_bias,
      1e4*forget_gate_hh_mtx, 1e4*forget_gate_hi_mtx, 1e4*forget_gate_bias,
      1e4*new_cell_hh_mtx, 1e4*new_cell_hi_mtx, 1e4*new_cell_bias,
      1e4*softmax_mtx, 1e4*softmax_bias)

if __name__ == '__main__':
  get_dyckkm_lstm_mlogk_params(k=4,m=3)

def get_dyckkm_lstm_mk_params(k=5,m=3):
  vocab = utils.get_vocab_of_bracket_types(k)[0]
  vocab_size = len(vocab)
  open_bracket_indices = list(range(0,k))
  close_bracket_indices = list(range(k,2*k))
  end_symbol_index = 2*k
  hidden_size = k*m
  num_stack_states = m

  # Vocabulary
  embedding_weights = torch.eye(vocab_size, vocab_size)

  # Input gate
  input_gate_hi_mtx = torch.stack(tuple((sum([embedding_weights[i] for i in open_bracket_indices]) for _ in range(hidden_size))))
  input_gate_hh_rows = []
  input_gate_bias_vals = []
  for i in range(m):
    if i == 0:
      neg_ones = -1*torch.ones(k*num_stack_states)
      row = neg_ones
    else:
      ones = torch.ones(k)
      neg_ones = -1*ones
      zeros_initial = torch.zeros(max(k*(i-1),0))
      zeros_final = torch.zeros(max(k*(num_stack_states-2-max(i-1,0)),0))
      row = torch.cat((zeros_initial, ones, neg_ones, zeros_final))
    input_gate_hh_rows.extend([row for _ in range(k)])
  input_gate_hh_mtx = torch.stack(input_gate_hh_rows)
  input_gate_bias = torch.tensor([-0.5 if i < k else -1.5 for i in range(hidden_size)])
  print('input')
  print(input_gate_hi_mtx)
  print(input_gate_bias)
  print(input_gate_hh_mtx)

  # Forget gate
  forget_gate_hi_mtx = -torch.stack(tuple((sum([embedding_weights[i] for i in close_bracket_indices]) for _ in range(hidden_size))))
  forget_gate_hh_rows = []
  for i in range(num_stack_states):
    neg_ones = -1*torch.ones(k)
    zeros_initial = torch.zeros(k*i)
    zeros_final = torch.zeros(max(k*(num_stack_states-1-i), 0))
    row = torch.cat((zeros_initial, neg_ones, zeros_final))
    forget_gate_hh_rows.extend([row for _ in range(k)])
  forget_gate_hh_mtx = torch.stack(forget_gate_hh_rows)
  forget_gate_bias = torch.ones(hidden_size)*0.5*TANH_OF_1 + 1
  print('forget')
  print(forget_gate_bias)
  print(forget_gate_hh_mtx)
  print(forget_gate_hi_mtx)

  # Output gate
  output_gate_hi_mtx = torch.stack(tuple((sum((embedding_weights[i] for i in close_bracket_indices)) for _ in range(hidden_size))))
  output_gate_hh_rows = []
  for i in range(num_stack_states):
    neg_max = -num_stack_states*torch.ones(k*max(i-1, 0))
    neg_ones = -torch.ones(k*min(i+1,2))
    zeros = torch.zeros(k*(num_stack_states-i-1))
    row = torch.cat((neg_max, neg_ones, zeros))
    output_gate_hh_rows.extend([row for _ in range(k)])
  output_gate_hh_mtx = torch.stack(output_gate_hh_rows).transpose(0,1)
  output_gate_bias = 0.5*torch.ones(hidden_size)
  print('output')
  print(output_gate_bias)
  print(output_gate_hh_mtx)




  # New cell candidate
  new_cell_hi_mtx = torch.stack(tuple(itertools.chain.from_iterable((embedding_weights[i] for i in open_bracket_indices) for x in range(num_stack_states))))
  new_cell_hh_mtx = torch.zeros(hidden_size, hidden_size)
  new_cell_bias = torch.zeros(hidden_size)
  print('new cell')
  print(new_cell_bias)
  print(new_cell_hh_mtx)
  print(new_cell_hi_mtx)

  # Softmax
  softmax_mtx = torch.zeros(len(vocab), hidden_size).float()
  softmax_bias = torch.zeros(len(vocab))

  # close-i
  for i,index_of_i in enumerate(close_bracket_indices):
    for j in range(num_stack_states):
      softmax_mtx[index_of_i,k*j+ i] = 1
    softmax_bias[index_of_i] = -TANH_OF_1*0.5
  # open-i
  for i, index_of_i in enumerate(open_bracket_indices):
    for j in range(k):
      softmax_mtx[index_of_i,k*(m-1)+j] = -1
    softmax_bias[index_of_i] = 0.5*TANH_OF_1
  # end
  for i in range(k):
    for j in range(num_stack_states):
      softmax_mtx[end_symbol_index,k*j+i] = -1
  softmax_bias[end_symbol_index]= 0.5*TANH_OF_1
  print('softmax')
  print(softmax_mtx)

  return (embedding_weights, 1e4*input_gate_hh_mtx, 1e4*input_gate_hi_mtx, 1e4*input_gate_bias,
      1e4*output_gate_hh_mtx, 1e4*output_gate_hi_mtx, 1e4*output_gate_bias,
      1e4*forget_gate_hh_mtx, 1e4*forget_gate_hi_mtx, 1e4*forget_gate_bias,
      1e4*new_cell_hh_mtx, 1e4*new_cell_hi_mtx, 1e4*new_cell_bias,
      1e4*softmax_mtx, 1e4*softmax_bias)
  




def get_mbounded_dyck2_writeverywhere_lstm_params(args={'lm':{'hidden_dim':8}}):
  hidden_size = args['lm']['hidden_dim']

  # Vocabulary
  embedding_weight_components = torch.zeros(6, hidden_size)
  embedding_weight_components[0][0] = 1
  embedding_weight_components[1][1] = 1
  embedding_weight_components[2][2] = 1
  embedding_weight_components[3][3] = 1
  embedding_weight_components[4][4] = 1
  embedding_weight_components[5][5] = 1

  open_a_vec = embedding_weight_components[0,:]
  open_b_vec = embedding_weight_components[1,:]
  end_vec = embedding_weight_components[4,:]
  start_vec = embedding_weight_components[5,:]
  close_a_vec = embedding_weight_components[2,:]
  close_b_vec = embedding_weight_components[3,:]
  embedding_weights = embedding_weight_components

  assert hidden_size % 2 == 0
  num_stack_states = hidden_size//2

  # Input gate
  input_gate_hi_mtx = torch.stack(tuple((open_a_vec + open_b_vec for _ in range(hidden_size))))
  input_gate_hh_rows = []
  input_gate_bias_vals = []
  for i in range(num_stack_states):
    if i == 0:
      neg_ones = -1*torch.ones(2*num_stack_states)
      row = neg_ones
    else:
      ones = torch.ones(2)
      neg_ones = -1*ones
      zeros_initial = torch.zeros(max(2*(i-1),0))
      zeros_final = torch.zeros(max(2*(num_stack_states-2-max(i-1,0)),0))
      row = torch.cat((zeros_initial, ones, neg_ones, zeros_final))
    input_gate_hh_rows.extend([row,row])
    input_gate_bias_vals.extend([-i-0.5, -i-0.5])
  input_gate_hh_mtx = torch.stack(input_gate_hh_rows)
  input_gate_bias = torch.tensor([-0.5 if i < 2 else -1.5 for i in range(hidden_size)])
  print('input')
  print(input_gate_bias)
  print(input_gate_hh_mtx)


  # Forget gate
  forget_gate_hi_mtx = -torch.stack(tuple((close_a_vec + close_b_vec for _ in range(hidden_size))))
  forget_gate_hh_rows = []
  for i in range(num_stack_states):
    neg_ones = -1*ones
    zeros_initial = torch.zeros(2*i)
    zeros_final = torch.zeros(max(2*(num_stack_states-1-i),0))
    row = torch.cat((zeros_initial, neg_ones, zeros_final))
    forget_gate_hh_rows.extend([row,row])
  forget_gate_hh_mtx = torch.stack(forget_gate_hh_rows)
  forget_gate_bias = torch.ones(hidden_size)*0.5*TANH_OF_1 + 1
  print('forget')
  print(forget_gate_bias)
  print(forget_gate_hh_mtx)

  # Output gate
  output_gate_hi_mtx = torch.stack(tuple((close_a_vec + close_b_vec for _ in range(hidden_size))))
  output_gate_hh_rows = []
  for i in range(num_stack_states):
    neg_max = -num_stack_states*torch.ones(2*max(i-1, 0))
    neg_ones = -torch.ones(2*min(i+1,2))
    zeros = torch.zeros(2*(num_stack_states-i-1))
    row = torch.cat((neg_max, neg_ones, zeros))
    output_gate_hh_rows.extend((row, row))
  output_gate_hh_mtx = torch.stack(output_gate_hh_rows).transpose(0,1)
  output_gate_bias = 0.5*torch.ones(hidden_size)
  print('output')
  print(output_gate_bias)
  print(output_gate_hh_mtx)


  # New cell candidate
  new_cell_hi_mtx = torch.stack(tuple(itertools.chain.from_iterable((open_a_vec, open_b_vec)
    for x in range(num_stack_states))))
  new_cell_hh_mtx = torch.zeros(hidden_size, hidden_size)
  new_cell_bias = torch.zeros(hidden_size)
  print('new cell')
  print(new_cell_bias)
  print(new_cell_hh_mtx)

  # Softmax
  softmax_mtx = torch.zeros(len(vocab), hidden_size).float()
  softmax_bias = torch.zeros(len(vocab))
  # close-a
  for i in range(num_stack_states):
    softmax_mtx[vocab['a)'],2*i] = 1
    softmax_mtx[vocab['a)'],2*i+1] = -1
    softmax_mtx[vocab['END'],2*i] = -1
    softmax_mtx[vocab['END'],2*i+1] = -1
  softmax_bias[vocab['a)']] = -TANH_OF_1*.5
  # close-b
  softmax_mtx[vocab['b)']] = - softmax_mtx[vocab['a)']]
  softmax_bias[vocab['b)']] = -TANH_OF_1*.5
  # open-a
  softmax_mtx[vocab['(a'], -2] = -1
  softmax_mtx[vocab['(a'], -1] = -1
  softmax_bias[vocab['(a']] = 0.5*TANH_OF_1
  # open-b
  softmax_mtx[vocab['(b']] = softmax_mtx[vocab['(a']]
  softmax_bias[vocab['(b']] = softmax_bias[vocab['(a']]
  # end
  softmax_bias[vocab['END']] = 0.5*TANH_OF_1
  # start
  #softmax_bias[vocab['START']] = -1


  #return (embedding_weights, input_gate_hh_mtx, input_gate_hi_mtx, input_gate_bias,
  #    output_gate_hh_mtx, output_gate_hi_mtx, output_gate_bias,
  #    forget_gate_hh_mtx, forget_gate_hi_mtx, forget_gate_bias,
  #    new_cell_hh_mtx, new_cell_hi_mtx, new_cell_bias,
  #    softmax_mtx, softmax_bias)
  return (embedding_weights, 1e4*input_gate_hh_mtx, 1e4*input_gate_hi_mtx, 1e4*input_gate_bias,
      1e4*output_gate_hh_mtx, 1e4*output_gate_hi_mtx, 1e4*output_gate_bias,
      1e4*forget_gate_hh_mtx, 1e4*forget_gate_hi_mtx, 1e4*forget_gate_bias,
      1e4*new_cell_hh_mtx, 1e4*new_cell_hi_mtx, 1e4*new_cell_bias,
      1e4*softmax_mtx, 1e4*softmax_bias)

