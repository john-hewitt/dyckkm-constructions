import torch
import utils
import torch
import itertools
from torch import nn
import math

SCALE_ALPHA = 1e4

def get_efficient_ctilde_u_mtx(k):
  """
  Constructs the efficient O(\log k) encoding
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
    vec = torch.cat((all_zeros_1, all_zeros_2, -all_neg_ones))
    softmax_vec = torch.cat((all_zeros_1, all_zeros_2, all_neg_ones))
    embeddings.append(vec)
    softmaxes.append(softmax_vec)
  embeddings = torch.stack(embeddings)
  softmaxes = torch.stack(softmaxes)
  return embeddings.transpose(0,1), softmaxes

def get_dyckkm_simplernn_mlogk_params(k,m):
  """
  Returns Simple RNN parameters for O(m\log k) construction
  generating Dyck-(k,m)
  """
  assert math.log2(k).is_integer()
  vocab = utils.get_vocab_of_bracket_types(k)[0]
  print(vocab)
  vocab_size = len(vocab)
  open_bracket_indices = list(range(0,k))
  close_bracket_indices = list(range(k,2*k))
  end_symbol_index = 2*k
  slot_size = 3*(int(math.log2(k)))-1
  #slot_size = k
  print('Slot size', 2*slot_size)
  hidden_size = slot_size*m
  print('hidden size', 2*hidden_size)
  num_stack_states = m

  # Vocabulary
  embedding_weights = torch.eye(vocab_size-1, vocab_size-1)

  # W (recurrent matrix)
  matrixDown = torch.FloatTensor([[1 if x==y-slot_size else 0 for x in range(slot_size*m)] for y in range(slot_size*m)])
  matrixUp = torch.FloatTensor([[1 if x==y+slot_size else 0 for x in range(slot_size*m)] for y in range(slot_size*m)])
  Wtop = torch.cat((matrixDown, matrixDown),1)
  Wbottom = torch.cat((matrixUp, matrixUp),1)
  W = 2*torch.cat((Wtop, Wbottom),0)
  print('W',W.shape, W)

  # U (input matrix)
  efficient_embeddings, efficient_softmaxes = get_efficient_ctilde_u_mtx(k)
  one_slot_k = torch.ones(slot_size,k)
  zero_mslot_k= torch.zeros((m-1)*slot_size,k)
  zero_slot_k = torch.zeros(slot_size, k)
  one_mslot_k = torch.ones((m-1)*slot_size, k)
  U = torch.cat((
      torch.cat((2*efficient_embeddings, -2*one_slot_k),1),
      torch.cat((zero_mslot_k, -2*one_mslot_k),1),
      torch.cat((-2*one_slot_k, zero_slot_k),1),
      torch.cat((-2*one_mslot_k, zero_mslot_k),1),
      ),0)
  print('U', U)

  # b (bias)
  bias = -torch.ones(hidden_size*2)
  print('b', bias)

  # softmax
  softmax_mtx = torch.zeros(len(vocab), hidden_size).float()
  softmax_bias = torch.zeros(len(vocab))

  # close-i
  for i, index_of_i in enumerate(close_bracket_indices):
    softmax_mtx[index_of_i,0:slot_size] = efficient_softmaxes[i]
    #softmax_mtx[index_of_i,i] = 1
    softmax_bias[index_of_i] = -0.5

  # open-i
  for i, index_of_i in enumerate(open_bracket_indices):
    for j in range(slot_size):
      softmax_mtx[index_of_i,slot_size*(m-1)+j] = -1
    softmax_bias[index_of_i] = 0.5

  # end
  for i in range(slot_size):
    for j in range(num_stack_states):
      softmax_mtx[end_symbol_index,slot_size*j+i] = -1
  softmax_bias[end_symbol_index]= 0.5
  softmax_mtx = torch.cat((softmax_mtx, softmax_mtx),1)
  #softmax_bias = torch.cat((softmax_bias, softmax_bias),0)
  print('softmax')
  print('softmax mtx', softmax_mtx)
  print('softmax bias', softmax_bias)

  return (embedding_weights, 1e4*W, 1e4*U, 1e4*bias, 1e4*softmax_mtx, 1e4*softmax_bias)

def get_dyckkm_simplernn_mk_params(k,m):
  """
  Returns Simple RNN parameters for O(mk) construction
  generating Dyck-(k,m)
  """

  assert math.log2(k).is_integer()
  vocab = utils.get_vocab_of_bracket_types(k)[0]
  print(vocab)
  vocab_size = len(vocab)
  open_bracket_indices = list(range(0,k))
  close_bracket_indices = list(range(k,2*k))
  end_symbol_index = 2*k
  #slot_size = 3*(int(math.log2(k)))-1
  slot_size = k
  print('Slot size', slot_size)
  hidden_size = slot_size*m
  print('hidden size', hidden_size)
  num_stack_states = m

  # Vocabulary
  embedding_weights = torch.eye(vocab_size-1, vocab_size-1)

  # W (recurrent matrix)
  matrixDown = torch.FloatTensor([[1 if x==y-k else 0 for x in range(k*m)] for y in range(k*m)])
  matrixUp = torch.FloatTensor([[1 if x==y+k else 0 for x in range(k*m)] for y in range(k*m)])
  Wtop = torch.cat((matrixDown, matrixDown),1)
  Wbottom = torch.cat((matrixUp, matrixUp),1)
  W = 2*torch.cat((Wtop, Wbottom),0)
  print('W', W)

  # U (input matrix)
  #encodings = torch.stack(tuple((sum([embedding_weights[i] for i in open_bracket_indices]) for _ in range(slot_size))))
  encodings = torch.eye(slot_size)
  one_slot_slot = torch.ones(slot_size,slot_size)
  zero_mslot_slot = torch.zeros((m-1)*slot_size,slot_size)
  zero_slot_slot = torch.zeros(slot_size, slot_size)
  one_mslot_slot = torch.ones((m-1)*slot_size, slot_size)
  U = torch.cat((
      torch.cat((2*encodings, -2*one_slot_slot),1),
      torch.cat((zero_mslot_slot, -2*one_mslot_slot),1),
      torch.cat((-2*one_slot_slot, zero_slot_slot),1),
      torch.cat((-2*one_mslot_slot, zero_mslot_slot),1),
      ),0)
      #print(a.shape,b.shape,c.shape,d.shape)
  print('U', U)

  # b (bias)
  bias = -torch.ones(hidden_size*2)
  print('b', bias)

  # softmax
  softmax_mtx = torch.zeros(len(vocab), hidden_size).float()
  softmax_bias = torch.zeros(len(vocab))

  # close-i
  for i, index_of_i in enumerate(close_bracket_indices):
    softmax_mtx[index_of_i,i] = 1
    softmax_bias[index_of_i] = -0.5

  # open-i
  for i, index_of_i in enumerate(open_bracket_indices):
    for j in range(k):
      softmax_mtx[index_of_i,slot_size*(m-1)+j] = -1
    softmax_bias[index_of_i] = 0.5

  # end
  for i in range(k):
    for j in range(num_stack_states):
      softmax_mtx[end_symbol_index,slot_size*j+i] = -1
  softmax_bias[end_symbol_index]= 0.5
  softmax_mtx = torch.cat((softmax_mtx, softmax_mtx),1)
  #softmax_bias = torch.cat((softmax_bias, softmax_bias),0)
  print('softmax')
  print('softmax mtx', softmax_mtx)
  print('softmax bias', softmax_bias)

  return (embedding_weights, 1e4*W, 1e4*U, 1e4*bias, 1e4*softmax_mtx, 1e4*softmax_bias)

if __name__ == '__main__':
  get_dyckkm_simplernn_mk_params(2,3)
  
